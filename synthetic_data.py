"""
Stat 6130 Project - Synthetic Data Analysis
Comparison of PCA, MDS, and Autoencoder dimensionality reduction
on synthetic datasets: Linear Subspace, S-Curve, Swiss Roll
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

N_SAMPLES = 1000
LATENT_DIM = 10       
VIZ_DIM = 2          
N_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

HP_HIDDEN_DIMS = [(64,), (128, 64), (256, 128), (128, 64, 32)]
HP_LEARNING_RATES = [1e-3, 3e-4]

def generate_linear_subspace(n_samples=N_SAMPLES, ambient_dim=20, noise_std=0.5):
    Z = np.random.randn(n_samples, 2)              
    A = np.random.randn(2, ambient_dim)             
    X = Z @ A                                       
    X += np.random.randn(n_samples, ambient_dim) * noise_std
    return X, Z


def generate_s_curve(n_samples=N_SAMPLES):
    X, color = make_s_curve(n_samples=n_samples, noise=0.1, random_state=RANDOM_SEED)
    return X, color


def generate_swiss_roll(n_samples=N_SAMPLES):
    X, color = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=RANDOM_SEED)
    return X, color

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=(128, 64)):
        super().__init__()

        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        encoder_layers += [nn.Linear(in_dim, latent_dim)]
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        decoder_layers += [nn.Linear(in_dim, input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X, latent_dim, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE, hidden_dims=(128, 64)):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim,
                        hidden_dims=hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    return model


def autoencoder_encode_decode(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        Z = model.encode(X_t).numpy()
        X_recon = model(X_t).numpy()
    return Z, X_recon


def tune_autoencoder_hyperparams(X_scaled, latent_dim, val_frac=0.2):
    X_tr, X_val = train_test_split(X_scaled, test_size=val_frac,
                                   random_state=RANDOM_SEED)
    input_dim = X_scaled.shape[1]
    max_hidden = max(input_dim * 2, latent_dim * 4)

    best_val_mse = np.inf
    best_config = None
    results = []

    for hidden_dims in HP_HIDDEN_DIMS:
        capped = tuple(min(h, max_hidden) for h in hidden_dims)
        for lr in HP_LEARNING_RATES:
            model = train_autoencoder(X_tr, latent_dim, hidden_dims=capped, lr=lr)
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                X_recon = model(X_val_t).numpy()
            val_mse = float(np.mean((X_val - X_recon) ** 2))
            results.append({"hidden_dims": hidden_dims, "capped": capped,
                            "lr": lr, "val_mse": val_mse})
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config = results[-1]

    return best_config, results

def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_train)
    X_recon = pca.inverse_transform(Z)
    return Z, X_recon, pca


def apply_mds(X, n_components):
    mds = MDS(n_components=n_components, random_state=RANDOM_SEED,
              normalized_stress="auto", n_jobs=-1)
    Z = mds.fit_transform(X)
    W = np.linalg.lstsq(X, Z, rcond=None)[0]  
    X_recon = Z @ W.T
    return Z, X_recon


def run_all_methods(X, latent_dim=LATENT_DIM, hidden_dims=(128, 64),
                    ae_lr=LEARNING_RATE):
    latent_dim = min(latent_dim, X.shape[1] - 1)
    max_hidden = max(X.shape[1] * 2, latent_dim * 4)
    hidden_dims = tuple(min(h, max_hidden) for h in hidden_dims)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    Z_pca, X_pca_recon, _ = apply_pca(X_scaled, X_scaled, latent_dim)
    X_pca_recon_orig = scaler.inverse_transform(X_pca_recon)

    # MDS
    Z_mds, X_mds_recon = apply_mds(X_scaled, latent_dim)
    X_mds_recon_orig = scaler.inverse_transform(X_mds_recon)

    # Autoencoder
    ae_model = train_autoencoder(X_scaled, latent_dim, hidden_dims=hidden_dims,
                                 lr=ae_lr)
    Z_ae, X_ae_recon = autoencoder_encode_decode(ae_model, X_scaled)
    X_ae_recon_orig = scaler.inverse_transform(X_ae_recon)

    embeddings = {"PCA": Z_pca, "MDS": Z_mds, "Autoencoder": Z_ae}
    reconstructions = {
        "PCA": X_pca_recon_orig,
        "MDS": X_mds_recon_orig,
        "Autoencoder": X_ae_recon_orig,
    }
    return embeddings, reconstructions, scaler

def compute_reconstruction_mse(X_original, reconstructions):
    mse = {}
    for method, X_recon in reconstructions.items():
        errors = np.mean((X_original - X_recon) ** 2, axis=1)   # per-sample MSE
        mse[method] = errors
    return mse


def hypothesis_tests(mse_dict):
    methods = list(mse_dict.keys())
    results = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            stat, p = stats.wilcoxon(mse_dict[a], mse_dict[b])
            results[f"{a} vs {b}"] = {"statistic": stat, "p_value": p}
    return results


def print_results(dataset_name, mse_dict, test_results):
    print(f"Dataset: {dataset_name}")
    print(f"\nReconstruction MSE (10 dimensions):")
    print(f"  {'Method':<15} {'Mean MSE':>12} {'Std MSE':>12}")
    for method, errors in mse_dict.items():
        print(f"  {method:<15} {errors.mean():>12.4f} {errors.std():>12.4f}")

    print(f"\nPairwise Wilcoxon signed-rank tests (H0: equal MSE):")
    print(f"  {'Comparison':<30} {'Statistic':>12} {'p-value':>12}")
    for comparison, res in test_results.items():
        sig = " *" if res["p_value"] < 0.05 else ""
        print(f"  {comparison:<30} {res['statistic']:>12.1f} {res['p_value']:>12.4f}{sig}")

def visualize_2d(dataset_name, X, color, embeddings_2d):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f"{dataset_name} — 2D Visualizations", fontsize=14, fontweight="bold")
    ax = axes[0]
    if X.shape[1] >= 3:
        ax.scatter(X[:, 0], X[:, 1], c=color, cmap="Spectral", s=8, alpha=0.7)
        ax.set_title("Original (dim 1 vs 2)")
    else:
        ax.scatter(X[:, 0], X[:, 1], c=color, cmap="Spectral", s=8, alpha=0.7)
        ax.set_title("Original")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    method_colors = {"PCA": "viridis", "MDS": "plasma", "Autoencoder": "inferno"}
    for ax, (method, Z) in zip(axes[1:], embeddings_2d.items()):
        ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=method_colors[method], s=8, alpha=0.7)
        ax.set_title(method)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"{dataset_name.lower().replace(' ', '_')}_2d.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()


def visualize_mse_distributions(dataset_name, mse_dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [errors for errors in mse_dict.values()]
    labels = list(mse_dict.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"{dataset_name} — Per-Sample Reconstruction MSE (10D)", fontsize=12)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Method")
    plt.tight_layout()
    fname = os.path.join(FIG_DIR, f"{dataset_name.lower().replace(' ', '_')}_mse.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()

def run_dataset(name, X, color):
    print(f"\nProcessing: {name}  (shape={X.shape})")

    print("  Tuning autoencoder hyperparameters...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latent_dim_capped = min(LATENT_DIM, X.shape[1] - 1)
    best_config, all_hp_results = tune_autoencoder_hyperparams(X_scaled, latent_dim_capped)
    best_hidden = best_config["hidden_dims"]
    best_lr = best_config["lr"]
    print(f"  Best config: hidden_dims={best_hidden}, lr={best_lr:.0e}, "
          f"val_mse={best_config['val_mse']:.4f}")
    print("  HP search results:")
    for r in sorted(all_hp_results, key=lambda x: x["val_mse"]):
        print(f"    hidden={r['hidden_dims']}, lr={r['lr']:.0e}  ->  val_mse={r['val_mse']:.4f}")

    print("  Training models (10D)...")
    embeddings_10d, reconstructions_10d, _ = run_all_methods(
        X, latent_dim=LATENT_DIM, hidden_dims=best_hidden, ae_lr=best_lr
    )
    mse_dict = compute_reconstruction_mse(X, reconstructions_10d)
    test_results = hypothesis_tests(mse_dict)
    print_results(name, mse_dict, test_results)

    print("  Training models (2D)...")
    embeddings_2d, _, _ = run_all_methods(
        X, latent_dim=VIZ_DIM, hidden_dims=best_hidden, ae_lr=best_lr
    )
    visualize_2d(name, X, color, embeddings_2d)
    visualize_mse_distributions(name, mse_dict)

    return mse_dict, test_results, best_config



def main():

    X_linear, Z_linear = generate_linear_subspace()
    color_linear = Z_linear[:, 0]

    X_scurve, color_scurve = generate_s_curve()

    X_swiss, color_swiss = generate_swiss_roll()

    datasets = [
        ("Linear Subspace", X_linear, color_linear),
        ("S-Curve",         X_scurve,  color_scurve),
        ("Swiss Roll",      X_swiss,   color_swiss),
    ]

    all_results = {}
    for name, X, color in datasets:
        mse_dict, test_results, best_config = run_dataset(name, X, color)
        all_results[name] = {"mse": mse_dict, "tests": test_results,
                             "best_hp": best_config}
    print("SUMMARY — Mean Reconstruction MSE (10D)")
    print(f"{'Dataset':<20} {'PCA':>10} {'MDS':>10} {'Autoencoder':>12}  {'Best AE Config'}")
    for name, res in all_results.items():
        pca_mse  = res["mse"]["PCA"].mean()
        mds_mse  = res["mse"]["MDS"].mean()
        ae_mse   = res["mse"]["Autoencoder"].mean()
        hp = res["best_hp"]
        print(f"{name:<20} {pca_mse:>10.4f} {mds_mse:>10.4f} {ae_mse:>12.4f}"
              f"  hidden={hp['hidden_dims']}, lr={hp['lr']:.0e}")


if __name__ == "__main__":
    main()
