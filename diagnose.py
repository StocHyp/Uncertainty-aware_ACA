"""
Diagnose a trained transport map by comparing its samples to the
original MCMC samples.

Produces:
- 1-D marginal histograms (one panel per dimension), all on a single PNG
- Summary statistics table comparing means, stds, quantiles
- Correlation matrix difference (heatmap + max abs delta)
- Kolmogorov-Smirnov statistic per dimension
- Pass/fail verdict at the end

Usage:
    # First generate samples to compare:
    python sample.py --checkpoint ./out/transport_map.pt \
                     --n_samples 200000 --out gen.npy

    # Then diagnose:
    python diagnose.py --mcmc thinned.txt --gen gen.npy --out_dir ./diag
"""

import argparse
from pathlib import Path

import numpy as np

from utils import load_samples, resolve_delimiter

# Lazy import of matplotlib so the script gives a clear message if it's
# missing rather than failing on `import diagnose` at the top
def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ks_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic (no scipy dependency)."""
    a = np.sort(a)
    b = np.sort(b)
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcmc", required=True,
                   help="Original (thinned) MCMC samples file")
    p.add_argument("--gen", required=True,
                   help="Generated samples from sample.py")
    p.add_argument("--mcmc_delimiter", default=None)
    p.add_argument("--gen_delimiter", default=None)
    p.add_argument("--out_dir", default="./diag")
    p.add_argument("--max_points", type=int, default=100_000,
                   help="Subsample each set down to this many rows for "
                        "histograms (full set still used for moments/KS)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()

    mcmc = load_samples(args.mcmc,
                        delimiter=resolve_delimiter(args.mcmc_delimiter))
    gen = load_samples(args.gen,
                       delimiter=resolve_delimiter(args.gen_delimiter))
    if mcmc.shape[1] != gen.shape[1]:
        raise ValueError(
            f"dimension mismatch: mcmc {mcmc.shape}, gen {gen.shape}")
    dim = mcmc.shape[1]
    print(f"[info] mcmc: {mcmc.shape}, gen: {gen.shape}")

    # Subsample for plot speed (moments/KS still use full)
    rng = np.random.default_rng(args.seed)
    if mcmc.shape[0] > args.max_points:
        mcmc_plot = mcmc[rng.choice(mcmc.shape[0], args.max_points, replace=False)]
    else:
        mcmc_plot = mcmc
    if gen.shape[0] > args.max_points:
        gen_plot = gen[rng.choice(gen.shape[0], args.max_points, replace=False)]
    else:
        gen_plot = gen

    # ------------ Sanity: are the magnitudes even comparable? ------------ #
    # If the gen samples are wildly out of MCMC range, the rest of the
    # diagnostics will be useless. Bail with a clear message.
    mcmc_max_abs = np.abs(mcmc).max()
    gen_max_abs = np.abs(gen).max()
    print(f"[info] max |value|: mcmc={mcmc_max_abs:.3g}, gen={gen_max_abs:.3g}")
    if not np.all(np.isfinite(gen)):
        print("[FAIL] generated samples contain NaN/Inf. Stop here.")
        return
    if gen_max_abs > 100 * max(mcmc_max_abs, 1e-12):
        print("[FAIL] generated samples are >100x larger than MCMC range.")
        print("       The model is overfit; retrain with stronger "
              "regularization (smaller architecture, more weight decay, "
              "or thin the chain more).")
        return

    # ------------ Per-dim summary statistics ------------ #
    stats_rows = []
    for j in range(dim):
        m_mean, g_mean = mcmc[:, j].mean(), gen[:, j].mean()
        m_std,  g_std  = mcmc[:, j].std(),  gen[:, j].std()
        # Standardize "error" by mcmc std so each dim is comparable
        scale = max(m_std, 1e-12)
        mean_err = (g_mean - m_mean) / scale
        std_err = (g_std - m_std) / scale
        ks = ks_1d(mcmc[:, j], gen[:, j])
        stats_rows.append((j, m_mean, g_mean, m_std, g_std, mean_err, std_err, ks))

    stats_path = out_dir / "stats.txt"
    with open(stats_path, "w") as f:
        header = ("dim    mcmc_mean      gen_mean   mcmc_std    gen_std  "
                  "mean_err/sd  std_err/sd      KS\n")
        f.write(header)
        for r in stats_rows:
            f.write(f"{r[0]:3d}  {r[1]:11.4g}  {r[2]:11.4g}  "
                    f"{r[3]:9.4g}  {r[4]:9.4g}  "
                    f"{r[5]:11.4f}  {r[6]:11.4f}  {r[7]:7.4f}\n")
    print(f"[info] wrote per-dim stats -> {stats_path}")

    mean_errs = np.array([r[5] for r in stats_rows])
    std_errs = np.array([r[6] for r in stats_rows])
    kss = np.array([r[7] for r in stats_rows])

    print(f"[info] mean error / sigma:  median={np.median(np.abs(mean_errs)):.3f}  "
          f"max={np.max(np.abs(mean_errs)):.3f}")
    print(f"[info] std  error / sigma:  median={np.median(np.abs(std_errs)):.3f}  "
          f"max={np.max(np.abs(std_errs)):.3f}")
    print(f"[info] KS statistic:        median={np.median(kss):.3f}  "
          f"max={np.max(kss):.3f}")

    # ------------ Correlation matrix comparison ------------ #
    C_m = np.corrcoef(mcmc.T)
    C_g = np.corrcoef(gen.T)
    dC = C_g - C_m
    max_abs_dcorr = float(np.max(np.abs(dC)))
    rms_dcorr = float(np.sqrt(np.mean(dC ** 2)))
    print(f"[info] correlation diff:    max|dC|={max_abs_dcorr:.4f}  "
          f"RMS={rms_dcorr:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(C_m, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title("MCMC correlation")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(C_g, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_title("MAF correlation")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    vmax = max(0.05, max_abs_dcorr)
    im2 = axes[2].imshow(dC, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[2].set_title(f"diff (max|dC|={max_abs_dcorr:.3f})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "correlations.png", dpi=110)
    plt.close(fig)
    print(f"[info] wrote correlation plot -> {out_dir / 'correlations.png'}")

    # ------------ 1-D marginal histograms ------------ #
    # Use MCMC range for x-axis so generated outliers don't squash the plot
    n_cols = 5
    n_rows = (dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    for j in range(dim):
        ax = axes[j]
        lo, hi = np.quantile(mcmc[:, j], [0.001, 0.999])
        pad = 0.1 * (hi - lo + 1e-12)
        x_lo, x_hi = lo - pad, hi + pad
        bins = np.linspace(x_lo, x_hi, 80)
        ax.hist(mcmc_plot[:, j], bins=bins, alpha=0.5, density=True,
                label="MCMC", color="C0")
        ax.hist(np.clip(gen_plot[:, j], x_lo, x_hi), bins=bins, alpha=0.5,
                density=True, label="MAF", color="C1")
        ax.set_title(f"dim {j}  KS={kss[j]:.2f}", fontsize=9)
        ax.set_xlim(x_lo, x_hi)
        ax.tick_params(labelsize=7)
    for j in range(dim, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "marginals.png", dpi=110)
    plt.close(fig)
    print(f"[info] wrote marginal plot -> {out_dir / 'marginals.png'}")

    # ------------ Verdict ------------ #
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    pass_mean  = np.max(np.abs(mean_errs)) < 0.05
    pass_std   = np.max(np.abs(std_errs))  < 0.10
    pass_ks    = np.max(kss)               < 0.05
    pass_corr  = max_abs_dcorr             < 0.05

    def mark(b): return "PASS" if b else "FAIL"
    print(f"  per-dim mean error < 0.05 sigma  ... {mark(pass_mean)}  "
          f"(max {np.max(np.abs(mean_errs)):.3f})")
    print(f"  per-dim std  error < 0.10 sigma  ... {mark(pass_std)}  "
          f"(max {np.max(np.abs(std_errs)):.3f})")
    print(f"  per-dim KS         < 0.05        ... {mark(pass_ks)}  "
          f"(max {np.max(kss):.3f})")
    print(f"  max |corr diff|    < 0.05        ... {mark(pass_corr)}  "
          f"({max_abs_dcorr:.3f})")
    print()
    if pass_mean and pass_std and pass_ks and pass_corr:
        print("  Overall: GOOD. The flow is a faithful generative model "
              "of your posterior.")
    else:
        print("  Overall: not yet good enough. Suggestions in order:")
        if not pass_corr:
            print("   - Increase --n_layers to 7 or 8 (correlations need depth)")
        if not pass_std:
            print("   - Increase capacity (--n_layers, --hidden_dims)")
        if not pass_mean:
            print("   - Train longer (raise --patience)")
        if not pass_ks:
            print("   - Increase capacity OR reduce regularization slightly")
    print("=" * 60)


if __name__ == "__main__":
    main()
