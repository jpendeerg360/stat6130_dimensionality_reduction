"""
Stat 6130 Project - COIL-20 Analysis
Comparison of PCA, MDS, and Autoencoder dimensionality reduction on the
COIL-20 processed dataset.

[MPS edits applied: model + tensors moved onto torch device so training
runs on Apple Silicon GPU when available, otherwise CPU.]
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[device] Using {device}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COIL_DIR = os.path.join(BASE_DIR, "data", "coil-20-proc")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
IMG_DIM = 128 * 128
N_OBJECTS = 20
N_POSES = 72
LATENT_DIM = 10
VIZ_DIM = 2
N_EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

HP_HIDDEN_DIMS = [(512,), (512, 256), (1024, 256), (512, 128), (1024, 256, 64)]
HP_LEARNING_RATES = [1e-3, 3e-4]
HP_CONV_CHANNELS = [(32, 64, 128, 256), (16, 32, 64, 128)]

OBJECT_CMAP = plt.get_cmap("tab20", N_OBJECTS)
POSE_CMAP   = "hsv"
METHOD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_coil20(data_dir=COIL_DIR):
    pattern = re.compile(r"obj(\d+)__(\d+)\.png")
    records = []
    for fname in os.listdir(data_dir):
        m = pattern.fullmatch(fname)
        if m:
            obj_id   = int(m.group(1))
            pose_idx = int(m.group(2))
            records.append((obj_id, pose_idx, os.path.join(data_dir, fname)))

    records.sort(key=lambda r: (r[0], r[1]))

    X          = np.zeros((len(records), IMG_DIM), dtype=np.float32)
    object_ids = np.zeros(len(records), dtype=int)
    pose_idxs  = np.zeros(len(records), dtype=int)

    for i, (obj_id, pose_idx, fpath) in enumerate(records):
        img = np.array(Image.open(fpath), dtype=np.float32) / 255.0
        X[i]          = img.ravel()
        object_ids[i] = obj_id
        pose_idxs[i]  = pose_idx
    print(f"Loaded {len(records)} images  shape={X.shape}")
    return X, object_ids, pose_idxs


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(1024, 256)):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class ConvAutoencoder(nn.Module):
    """
    Encoder: 4x stride-2 Conv2d (128->64->32->16->8) then Linear to latent_dim.
    Decoder: Linear then 4x stride-2 ConvTranspose2d back to 128x128.
    Accepts and returns flat (N, IMG_DIM) vectors matching the FC autoencoder interface.
    """
    def __init__(self, img_size=128, latent_dim=10, channels=(32, 64, 128, 256)):
        super().__init__()
        self._channels = list(channels)
        self._enc_spatial = img_size // (2 ** len(channels))
        self._flat_dim = channels[-1] * self._enc_spatial ** 2

        enc_layers = []
        in_ch = 1
        for out_ch in channels:
            enc_layers += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.encoder_fc = nn.Linear(self._flat_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, self._flat_dim)
        rev_ch = list(reversed(channels))
        dec_layers = []
        for i in range(len(rev_ch)):
            out_ch = rev_ch[i + 1] if i + 1 < len(rev_ch) else 1
            act = nn.ReLU() if out_ch != 1 else nn.Identity()
            dec_layers += [
                nn.ConvTranspose2d(rev_ch[i], out_ch, 3, stride=2, padding=1, output_padding=1),
                act,
            ]
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 128, 128)
        return self.encoder_fc(self.encoder_conv(x).flatten(1))

    def forward(self, x):
        flat_in = (x.dim() == 2)
        z = self.encode(x)
        h = self.decoder_fc(z).view(-1, self._channels[-1], self._enc_spatial, self._enc_spatial)
        out = self.decoder_conv(h)
        return out.flatten(1) if flat_in else out


def train_autoencoder(X, latent_dim, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE, hidden_dims=(1024, 256)):
    X_t    = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
    model  = Autoencoder(X.shape[1], latent_dim, hidden_dims).to(device) 
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device)                                     
            opt.zero_grad()
            crit(model(batch), batch).backward()
            opt.step()
    return model


def train_conv_autoencoder(X, latent_dim, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                            lr=LEARNING_RATE, channels=(32, 64, 128, 256)):
    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
    model = ConvAutoencoder(img_size=128, latent_dim=latent_dim,
                            channels=channels).to(device)              
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for (batch,) in loader:
            batch = batch.to(device)                                
            opt.zero_grad()
            crit(model(batch), batch).backward()
            opt.step()
    return model


def ae_encode_decode(model, X):
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32).to(device)           
        Z     = model.encode(X_t).cpu().numpy()                     
        X_rec = model(X_t).cpu().numpy()                      
    return Z, X_rec


def tune_autoencoder_hyperparams(X_scaled, latent_dim, val_frac=0.2):
    X_tr, X_val = train_test_split(X_scaled, test_size=val_frac,
                                   random_state=RANDOM_SEED)
    best_val_mse = np.inf
    best_config  = None
    results      = []

    for hidden_dims in HP_HIDDEN_DIMS:
        for lr in HP_LEARNING_RATES:
            model = train_autoencoder(X_tr, latent_dim,
                                      hidden_dims=hidden_dims, lr=lr)
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device) 
                X_rec   = model(X_val_t).cpu().numpy()                       
            val_mse = float(np.mean((X_val - X_rec) ** 2))
            entry = {"hidden_dims": hidden_dims, "lr": lr, "val_mse": val_mse}
            results.append(entry)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config  = entry
    return best_config, results


def tune_conv_autoencoder_hyperparams(X_scaled, latent_dim, val_frac=0.2):
    X_tr, X_val = train_test_split(X_scaled, test_size=val_frac,
                                   random_state=RANDOM_SEED)
    best_val_mse = np.inf
    best_config  = None
    results      = []

    for channels in HP_CONV_CHANNELS:
        for lr in HP_LEARNING_RATES:
            model = train_conv_autoencoder(X_tr, latent_dim, channels=channels, lr=lr)
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)   # MPS edit
                X_rec   = model(X_val_t).cpu().numpy()                          # MPS edit
            val_mse = float(np.mean((X_val - X_rec) ** 2))
            entry = {"channels": channels, "lr": lr, "val_mse": val_mse}
            results.append(entry)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config  = entry
    return best_config, results


def apply_pca(X_scaled, n_components):
    pca   = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Z     = pca.fit_transform(X_scaled)
    X_rec = pca.inverse_transform(Z)
    return Z, X_rec


def apply_mds(X_scaled, n_components):
    mds   = MDS(n_components=n_components, random_state=RANDOM_SEED,
                normalized_stress="auto", n_jobs=-1)
    Z     = mds.fit_transform(X_scaled)
    W     = np.linalg.lstsq(X_scaled, Z, rcond=None)[0]
    X_rec = Z @ W.T
    return Z, X_rec


def run_all_methods(X, latent_dim, hidden_dims=(1024, 256), ae_lr=LEARNING_RATE,
                    conv_channels=None, conv_ae_lr=LEARNING_RATE):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z_pca, X_pca_rec = apply_pca(X_scaled, latent_dim)
    Z_mds, X_mds_rec = apply_mds(X_scaled, latent_dim)

    ae_model         = train_autoencoder(X_scaled, latent_dim,
                                         hidden_dims=hidden_dims, lr=ae_lr)
    Z_ae,  X_ae_rec  = ae_encode_decode(ae_model, X_scaled)

    embeddings      = {"PCA": Z_pca,     "MDS": Z_mds,     "Autoencoder": Z_ae}
    reconstructions = {
        "PCA":         scaler.inverse_transform(X_pca_rec),
        "MDS":         scaler.inverse_transform(X_mds_rec),
        "Autoencoder": scaler.inverse_transform(X_ae_rec),
    }

    if conv_channels is not None:
        cae_model = train_conv_autoencoder(X_scaled, latent_dim,
                                           channels=conv_channels, lr=conv_ae_lr)
        Z_cae, X_cae_rec = ae_encode_decode(cae_model, X_scaled)
        embeddings["Conv-AE"] = Z_cae
        reconstructions["Conv-AE"] = scaler.inverse_transform(X_cae_rec)

    return embeddings, reconstructions, scaler


def compute_reconstruction_mse(X, reconstructions):
    return {m: np.mean((X - R) ** 2, axis=1) for m, R in reconstructions.items()}


def hypothesis_tests(mse_dict):
    methods = list(mse_dict.keys())
    results = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            stat, p = stats.wilcoxon(mse_dict[a], mse_dict[b])
            results[f"{a} vs {b}"] = {"statistic": stat, "p_value": p}
    return results


def print_results(mse_dict, test_results):
    print("COIL-20 Reconstruction MSE (10D)")
    print(f"  {'Method':<15} {'Mean MSE':>12} {'Std MSE':>12}")
    for method, errors in mse_dict.items():
        print(f"  {method:<15} {errors.mean():>12.6f} {errors.std():>12.6f}")
    print(f"\nPairwise Wilcoxon signed-rank tests:")
    print(f"  {'Comparison':<30} {'Statistic':>12} {'p-value':>12}")
    for cmp, res in test_results.items():
        sig = " *" if res["p_value"] < 0.05 else ""
        print(f"  {cmp:<30} {res['statistic']:>12.1f} {res['p_value']:>12.4f}{sig}")


def pose_recovery_spearman(Z_2d, object_ids, pose_idxs):
    r_values = []
    for obj in range(1, N_OBJECTS + 1):
        mask = object_ids == obj
        pts  = Z_2d[mask]
        pose = pose_idxs[mask]

        centroid = pts.mean(axis=0)
        angles   = np.arctan2(pts[:, 1] - centroid[1],
                              pts[:, 0] - centroid[0])
        r, _ = stats.spearmanr(angles, pose)
        r_values.append(abs(r))

    return np.array(r_values)


def circularity_scores(Z_2d, object_ids):
    scores = []
    for obj in range(1, N_OBJECTS + 1):
        mask = object_ids == obj
        pts  = Z_2d[mask]
        cov  = np.cov(pts.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))
        ratio = eigvals[0] / (eigvals[1] + 1e-12)
        scores.append(ratio)
    return np.array(scores)


def classification_accuracy(embeddings_10d, object_ids):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    for method, Z in embeddings_10d.items():
        scores = cross_val_score(
            SVC(kernel='rbf', C=10, gamma='scale'),
            Z, object_ids, cv=cv, scoring="accuracy"
        )
        results[method] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {method:<15} accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
    return results


def visualize_2d_by_object(embeddings_2d, object_ids, pose_idxs):
    n_methods = len(embeddings_2d)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    fig.suptitle("COIL-20 — 2D Embeddings Colored by Object ID",
                 fontsize=14, fontweight="bold")

    for ax, (method, Z) in zip(axes, embeddings_2d.items()):
        for obj in range(1, N_OBJECTS + 1):
            mask = object_ids == obj
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       color=OBJECT_CMAP(obj - 1), s=8, alpha=0.7,
                       label=f"obj{obj}")
        ax.set_title(method)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=OBJECT_CMAP(i), markersize=6)
               for i in range(N_OBJECTS)]
    axes[-1].legend(handles, [f"obj{i+1}" for i in range(N_OBJECTS)],
                    bbox_to_anchor=(1.01, 1), loc="upper left",
                    fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coil20_2d_by_object.png"), dpi=150, bbox_inches="tight")
    print("  Saved: coil20_2d_by_object.png")
    plt.close()


def visualize_circular_structure(embeddings_2d, object_ids, pose_idxs,
                                 showcase_objects=(1, 5, 10, 15)):
    methods = list(embeddings_2d.keys())
    n_obj   = len(showcase_objects)
    fig, axes = plt.subplots(len(methods), n_obj,
                              figsize=(4 * n_obj, 4 * len(methods)))
    fig.suptitle("COIL-20 — Circular Rotation Structure per Object\n"
                 "(color = pose angle 0°→355°, circular colormap)",
                 fontsize=13, fontweight="bold")

    true_angles = pose_idxs * 5

    for row, method in enumerate(methods):
        Z = embeddings_2d[method]
        for col, obj in enumerate(showcase_objects):
            ax   = axes[row, col]
            mask = object_ids == obj
            pts  = Z[mask]
            ang  = true_angles[mask]

            sc = ax.scatter(pts[:, 0], pts[:, 1],
                            c=ang, cmap=POSE_CMAP, vmin=0, vmax=360,
                            s=20, alpha=0.9)
            order   = np.argsort(ang)
            loop_x  = np.append(pts[order, 0], pts[order[0], 0])
            loop_y  = np.append(pts[order, 1], pts[order[0], 1])
            ax.plot(loop_x, loop_y, color="gray", linewidth=0.5, alpha=0.5)

            if row == 0:
                ax.set_title(f"Object {obj}", fontsize=10)
            if col == 0:
                ax.set_ylabel(method, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=POSE_CMAP,
                                norm=plt.Normalize(vmin=0, vmax=360))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, label="Pose angle (degrees)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coil20_circular_structure.png"), dpi=150, bbox_inches="tight")
    print("  Saved: coil20_circular_structure.png")
    plt.close()


def visualize_rotation_metrics(spearman_results, circularity_results):
    methods = list(spearman_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("COIL-20 — Circular Structure Metrics", fontsize=13,
                 fontweight="bold")
    colors = METHOD_COLORS[:len(methods)]

    ax = axes[0]
    means = [spearman_results[m].mean() for m in methods]
    stds  = [spearman_results[m].std()  for m in methods]
    bars  = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Spearman r| (embedding angle vs. true pose)")
    ax.set_title("Pose Recovery")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    means = [circularity_results[m].mean() for m in methods]
    stds  = [circularity_results[m].std()  for m in methods]
    bars  = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Circularity ratio (minor/major eigenvalue)")
    ax.set_title("Circularity of Per-Object Pose Trajectory")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coil20_rotation_metrics.png"), dpi=150, bbox_inches="tight")
    print("  Saved: coil20_rotation_metrics.png")
    plt.close()


def visualize_mse(mse_dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = list(mse_dict.values())
    labels = list(mse_dict.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], METHOD_COLORS[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("COIL-20 — Per-Sample Reconstruction MSE (10D)", fontsize=12)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coil20_mse.png"), dpi=150, bbox_inches="tight")
    print("  Saved: coil20_mse.png")
    plt.close()


def visualize_classification(clf_results):
    methods = list(clf_results.keys())
    means   = [clf_results[m]["mean"] for m in methods]
    stds    = [clf_results[m]["std"]  for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, means, yerr=stds, capsize=5,
                  color=METHOD_COLORS[:len(methods)], alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("COIL-20 — Object Classification from 10D Embedding\n"
                 "(SVM, RBF Kernel)", fontsize=11)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coil20_classification.png"), dpi=150, bbox_inches="tight")
    print("  Saved: coil20_classification.png")
    plt.close()


def main():
    print("Loading COIL-20...")
    X, object_ids, pose_idxs = load_coil20()

    print("\nTuning autoencoder hyperparameters (10D task)...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_config, all_hp = tune_autoencoder_hyperparams(X_scaled, LATENT_DIM)
    best_hidden = best_config["hidden_dims"]
    best_lr     = best_config["lr"]
    print(f"  Best config: hidden_dims={best_hidden}, lr={best_lr:.0e}, "
          f"val_mse={best_config['val_mse']:.6f}")
    print("  Full HP search (sorted by val_mse):")
    for r in sorted(all_hp, key=lambda x: x["val_mse"]):
        print(f"    hidden={r['hidden_dims']}, lr={r['lr']:.0e}  ->  {r['val_mse']:.6f}")

    print("\nTuning conv autoencoder hyperparameters (10D task)...")
    best_conv_config, all_conv_hp = tune_conv_autoencoder_hyperparams(X_scaled, LATENT_DIM)
    best_conv_channels = best_conv_config["channels"]
    best_conv_lr       = best_conv_config["lr"]
    print(f"  Best conv config: channels={best_conv_channels}, lr={best_conv_lr:.0e}, "
          f"val_mse={best_conv_config['val_mse']:.6f}")
    print("  Full conv HP search (sorted by val_mse):")
    for r in sorted(all_conv_hp, key=lambda x: x["val_mse"]):
        print(f"    channels={r['channels']}, lr={r['lr']:.0e}  ->  {r['val_mse']:.6f}")

    print("\nTraining all methods (10D)...")
    embeddings_10d, reconstructions_10d, _ = run_all_methods(
        X, LATENT_DIM, hidden_dims=best_hidden, ae_lr=best_lr,
        conv_channels=best_conv_channels, conv_ae_lr=best_conv_lr
    )
    mse_dict     = compute_reconstruction_mse(X, reconstructions_10d)
    test_results = hypothesis_tests(mse_dict)
    print_results(mse_dict, test_results)

    print("\nTraining all methods (2D)...")
    embeddings_2d, _, _ = run_all_methods(
        X, VIZ_DIM, hidden_dims=best_hidden, ae_lr=best_lr,
        conv_channels=best_conv_channels, conv_ae_lr=best_conv_lr
    )

    print("\nComputing circular rotation structure metrics...")
    spearman_by_method    = {}
    circularity_by_method = {}
    for method, Z in embeddings_2d.items():
        sr  = pose_recovery_spearman(Z, object_ids, pose_idxs)
        cir = circularity_scores(Z, object_ids)
        spearman_by_method[method]    = sr
        circularity_by_method[method] = cir
        print(f"  {method:<15}  pose recovery |r| = {sr.mean():.3f} ± {sr.std():.3f}"
              f"   circularity = {cir.mean():.3f} ± {cir.std():.3f}")

    print("\nClassification accuracy (5-fold CV, SVM RBF kernel):")
    clf_results = classification_accuracy(embeddings_10d, object_ids)

    print("\nGenerating figures...")
    visualize_2d_by_object(embeddings_2d, object_ids, pose_idxs)
    visualize_circular_structure(embeddings_2d, object_ids, pose_idxs)
    visualize_rotation_metrics(spearman_by_method, circularity_by_method)
    visualize_mse(mse_dict)
    visualize_classification(clf_results)

    print("\n" + "=" * 65)
    print("COIL-20 SUMMARY")
    print("=" * 65)
    print(f"Best FC-AE config:   hidden_dims={best_hidden}, lr={best_lr:.0e}")
    print(f"Best Conv-AE config: channels={best_conv_channels}, lr={best_conv_lr:.0e}")
    print(f"\n{'Method':<15} {'MSE (mean)':>12} {'Accuracy':>10} "
          f"{'Pose |r|':>10} {'Circularity':>12}")
    print("-" * 65)
    for m in embeddings_10d:
        print(f"{m:<15} {mse_dict[m].mean():>12.6f} "
              f"{clf_results[m]['mean']:>10.4f} "
              f"{spearman_by_method[m].mean():>10.3f} "
              f"{circularity_by_method[m].mean():>12.3f}")


if __name__ == "__main__":
    main()