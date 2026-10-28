"""
Stat 6130 Project - TCGA BRCA Gene Expression Analysis
Comparison of PCA, MDS, and Autoencoder dimensionality reduction on
TCGA Breast Invasive Carcinoma RNA-seq data (UCSC Xena HiSeqV2).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "HiSeqV2.gz")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
N_VAR_GENES = 2000      
LATENT_DIM  = 10
VIZ_DIM     = 2
N_EPOCHS    = 200
BATCH_SIZE  = 64
LEARNING_RATE = 1e-3

HP_HIDDEN_DIMS   = [(128,), (256,), (256, 64), (512, 128)]
HP_LEARNING_RATES = [1e-3, 3e-4]
HP_DROPOUT        = [0.0, 0.2]   

SAMPLE_TYPE = {"01": "Tumor", "11": "Normal", "06": "Metastatic"}

def load_tcga(path=DATA_PATH, n_var_genes=N_VAR_GENES):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    print(f"  Raw matrix: {df.shape[0]} genes x {df.shape[1]} samples")

    df = df.T                  

    type_codes = df.index.str.split("-").str[-1].str[:2]
    labels     = type_codes.map(SAMPLE_TYPE).fillna("Unknown").values
    print(f"  Sample types: { {t: (labels==t).sum() for t in ['Tumor','Normal','Metastatic']} }")

    gene_var = df.var(axis=0)
    top_genes = gene_var.nlargest(n_var_genes).index
    X_sub     = df[top_genes].values.astype(np.float32)
    print(f"  Selected top {n_var_genes} variable genes  ->  shape {X_sub.shape}")

    return X_sub, labels, top_genes.tolist()

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 hidden_dims=(256, 64), dropout=0.0):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X, latent_dim, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                      lr=LEARNING_RATE, hidden_dims=(256, 64), dropout=0.0):
    X_t    = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
    model  = Autoencoder(X.shape[1], latent_dim, hidden_dims, dropout)
    opt    = optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for (batch,) in loader:
            opt.zero_grad()
            crit(model(batch), batch).backward()
            opt.step()
    return model


def ae_encode_decode(model, X):
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32)
        Z     = model.encode(X_t).numpy()
        X_rec = model(X_t).numpy()
    return Z, X_rec

def tune_autoencoder_hyperparams(X_scaled, latent_dim, val_frac=0.2):
    X_tr, X_val = train_test_split(X_scaled, test_size=val_frac,
                                   random_state=RANDOM_SEED)
    best_val_mse = np.inf
    best_config  = None
    results      = []

    total = len(HP_HIDDEN_DIMS) * len(HP_LEARNING_RATES) * len(HP_DROPOUT)
    done  = 0
    for hidden_dims in HP_HIDDEN_DIMS:
        for lr in HP_LEARNING_RATES:
            for dropout in HP_DROPOUT:
                done += 1
                print(f"    [{done}/{total}] hidden={hidden_dims}, "
                      f"lr={lr:.0e}, dropout={dropout}", end=" ... ")
                model = train_autoencoder(X_tr, latent_dim,
                                          hidden_dims=hidden_dims,
                                          lr=lr, dropout=dropout)
                model.eval()
                with torch.no_grad():
                    X_val_t = torch.tensor(X_val, dtype=torch.float32)
                    X_rec   = model(X_val_t).numpy()
                val_mse = float(np.mean((X_val - X_rec) ** 2))
                print(f"val_mse={val_mse:.6f}")
                entry = {"hidden_dims": hidden_dims, "lr": lr,
                         "dropout": dropout, "val_mse": val_mse}
                results.append(entry)
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_config  = entry

    return best_config, results


def apply_pca(X_scaled, n_components):
    pca   = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Z     = pca.fit_transform(X_scaled)
    X_rec = pca.inverse_transform(Z)
    return Z, X_rec, pca


def apply_mds(X_scaled, n_components):
    mds   = MDS(n_components=n_components, random_state=RANDOM_SEED,
                normalized_stress="auto", n_jobs=-1)
    Z     = mds.fit_transform(X_scaled)
    W     = np.linalg.lstsq(X_scaled, Z, rcond=None)[0]
    X_rec = Z @ W.T
    return Z, X_rec


def run_all_methods(X, latent_dim, hidden_dims=(256, 64),
                    ae_lr=LEARNING_RATE, ae_dropout=0.0):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z_pca, X_pca_rec, pca_obj = apply_pca(X_scaled, latent_dim)
    Z_mds, X_mds_rec           = apply_mds(X_scaled, latent_dim)

    ae_model        = train_autoencoder(X_scaled, latent_dim,
                                        hidden_dims=hidden_dims,
                                        lr=ae_lr, dropout=ae_dropout)
    Z_ae, X_ae_rec  = ae_encode_decode(ae_model, X_scaled)

    embeddings      = {"PCA": Z_pca,     "MDS": Z_mds,     "Autoencoder": Z_ae}
    reconstructions = {
        "PCA":         scaler.inverse_transform(X_pca_rec),
        "MDS":         scaler.inverse_transform(X_mds_rec),
        "Autoencoder": scaler.inverse_transform(X_ae_rec),
    }
    return embeddings, reconstructions, scaler, pca_obj

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
    print("TCGA BRCA Reconstruction MSE (10D)")
    print(f"  {'Method':<15} {'Mean MSE':>12} {'Std MSE':>12}")
    for method, errors in mse_dict.items():
        print(f"  {method:<15} {errors.mean():>12.6f} {errors.std():>12.6f}")
    print(f"\nPairwise Wilcoxon signed-rank tests:")
    print(f"  {'Comparison':<30} {'Statistic':>12} {'p-value':>12}")
    for cmp, res in test_results.items():
        sig = " *" if res["p_value"] < 0.05 else ""
        print(f"  {cmp:<30} {res['statistic']:>12.1f} {res['p_value']:>12.4f}{sig}")
    print("  (* = significant at alpha=0.05)")

def binary_classification_accuracy(embeddings_10d, labels):
    mask = np.isin(labels, ["Tumor", "Normal"])
    y    = (labels[mask] == "Tumor").astype(int)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    for method, Z in embeddings_10d.items():
        Z_sub  = Z[mask]
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            Z_sub, y, cv=cv, scoring="accuracy"
        )
        results[method] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {method:<15} Tumor/Normal accuracy = "
              f"{scores.mean():.4f} ± {scores.std():.4f}")
    return results

def visualize_pca_variance(pca_full):
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_comp  = np.arange(1, len(cum_var) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_comp, cum_var, linewidth=1.5, color="#4C72B0")
    ax.fill_between(n_comp, cum_var, alpha=0.15, color="#4C72B0")

    for threshold, color in [(50, "#DD8452"), (80, "#55A868"), (95, "#C44E52")]:
        idx = np.searchsorted(cum_var, threshold)
        ax.axvline(idx + 1, color=color, linestyle="--", linewidth=1,
                   label=f"{threshold}% @ PC{idx+1}")
        ax.axhline(threshold, color=color, linestyle="--", linewidth=0.7)

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title(f"TCGA BRCA — PCA Explained Variance\n"
                 f"(top {N_VAR_GENES} variable genes)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(1, min(200, len(cum_var)))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tcga_pca_variance.png"), dpi=150, bbox_inches="tight")
    print("  Saved: tcga_pca_variance.png")
    plt.close()


def visualize_2d_by_type(embeddings_2d, labels):
    type_color = {"Tumor": "#4C72B0", "Normal": "#DD8452", "Metastatic": "#55A868"}
    fig, axes  = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("TCGA BRCA — 2D Embeddings by Sample Type",
                 fontsize=13, fontweight="bold")

    for ax, (method, Z) in zip(axes, embeddings_2d.items()):
        for stype, color in type_color.items():
            mask = labels == stype
            if mask.sum() == 0:
                continue
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       color=color, s=10, alpha=0.6, label=stype)
        ax.set_title(method)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    axes[-1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tcga_2d_by_type.png"), dpi=150, bbox_inches="tight")
    print("  Saved: tcga_2d_by_type.png")
    plt.close()


def visualize_mse(mse_dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = list(mse_dict.values())
    labels = list(mse_dict.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452", "#55A868"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"TCGA BRCA — Per-Sample Reconstruction MSE (10D)\n"
                 f"(top {N_VAR_GENES} variable genes)", fontsize=11)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tcga_mse.png"), dpi=150, bbox_inches="tight")
    print("  Saved: tcga_mse.png")
    plt.close()


def visualize_classification(clf_results):
    methods = list(clf_results.keys())
    means   = [clf_results[m]["mean"] for m in methods]
    stds    = [clf_results[m]["std"]  for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, means, yerr=stds, capsize=5,
                  color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("TCGA BRCA — Tumor vs. Normal Classification\n"
                 "from 10D Embedding (Logistic Regression)", fontsize=11)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tcga_classification.png"), dpi=150, bbox_inches="tight")
    print("  Saved: tcga_classification.png")
    plt.close()


def visualize_hp_search(all_hp_results, best_config):
    dropouts = sorted(set(r["dropout"] for r in all_hp_results))
    fig, axes = plt.subplots(1, len(dropouts),
                              figsize=(6 * len(dropouts), 4), sharey=True)
    if len(dropouts) == 1:
        axes = [axes]

    fig.suptitle("TCGA AE Hyperparameter Search — Validation MSE",
                 fontsize=12, fontweight="bold")

    all_mses = [r["val_mse"] for r in all_hp_results]
    vmin, vmax = min(all_mses), max(all_mses)

    for ax, drop in zip(axes, dropouts):
        sub = [r for r in all_hp_results if r["dropout"] == drop]
        hidden_opts = HP_HIDDEN_DIMS
        lr_opts     = HP_LEARNING_RATES
        grid = np.full((len(hidden_opts), len(lr_opts)), np.nan)
        for r in sub:
            ri = hidden_opts.index(r["hidden_dims"])
            ci = lr_opts.index(r["lr"])
            grid[ri, ci] = r["val_mse"]

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(lr_opts)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lr_opts])
        ax.set_yticks(range(len(hidden_opts)))
        ax.set_yticklabels([str(h) for h in hidden_opts])
        ax.set_xlabel("Learning Rate")
        ax.set_title(f"Dropout = {drop}")
        if ax is axes[0]:
            ax.set_ylabel("Hidden Dims")

        for ri in range(len(hidden_opts)):
            for ci in range(len(lr_opts)):
                val = grid[ri, ci]
                if not np.isnan(val):
                    is_best = (hidden_opts[ri] == best_config["hidden_dims"]
                               and lr_opts[ci]     == best_config["lr"]
                               and drop            == best_config["dropout"])
                    fw = "bold" if is_best else "normal"
                    ax.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                            fontsize=8, fontweight=fw,
                            color="black" if val < (vmin + vmax) / 2 else "white")

        plt.colorbar(im, ax=ax, label="Val MSE")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "tcga_hp_search.png"), dpi=150, bbox_inches="tight")
    print("  Saved: tcga_hp_search.png")
    plt.close()

def main():
    X, labels, gene_names = load_tcga()

    print("\nFitting full PCA for explained variance curve...")
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    pca_full = PCA(n_components=min(200, X.shape[1]), random_state=RANDOM_SEED)
    pca_full.fit(X_scaled_full)
    visualize_pca_variance(pca_full)

    print(f"\nTuning autoencoder hyperparameters ({N_VAR_GENES}-gene input, 10D)...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_config, all_hp = tune_autoencoder_hyperparams(X_scaled, LATENT_DIM)
    best_hidden  = best_config["hidden_dims"]
    best_lr      = best_config["lr"]
    best_dropout = best_config["dropout"]
    print(f"\n  Best config: hidden_dims={best_hidden}, lr={best_lr:.0e}, "
          f"dropout={best_dropout}, val_mse={best_config['val_mse']:.6f}")
    visualize_hp_search(all_hp, best_config)

    print("\nTraining all methods (10D)...")
    embeddings_10d, reconstructions_10d, _, _ = run_all_methods(
        X, LATENT_DIM,
        hidden_dims=best_hidden, ae_lr=best_lr, ae_dropout=best_dropout
    )
    mse_dict     = compute_reconstruction_mse(X, reconstructions_10d)
    test_results = hypothesis_tests(mse_dict)
    print_results(mse_dict, test_results)

    print("\nTraining all methods (2D)...")
    embeddings_2d, _, _, _ = run_all_methods(
        X, VIZ_DIM,
        hidden_dims=best_hidden, ae_lr=best_lr, ae_dropout=best_dropout
    )

    print("\nClassification accuracy (5-fold CV, Tumor vs. Normal):")
    clf_results = binary_classification_accuracy(embeddings_10d, labels)

    print("\nGenerating figures...")
    visualize_2d_by_type(embeddings_2d, labels)
    visualize_mse(mse_dict)
    visualize_classification(clf_results)

    print("TCGA BRCA SUMMARY")
    print(f"Genes used: top {N_VAR_GENES} by variance  |  "
          f"Best AE: hidden={best_hidden}, lr={best_lr:.0e}, "
          f"dropout={best_dropout}")
    print(f"\n{'Method':<15} {'MSE (mean)':>12} {'Accuracy':>10}")
    for m in ["PCA", "MDS", "Autoencoder"]:
        print(f"{m:<15} {mse_dict[m].mean():>12.6f} "
              f"{clf_results[m]['mean']:>10.4f}")


if __name__ == "__main__":
    main()
