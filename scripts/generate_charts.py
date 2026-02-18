"""
Generate publication-quality charts for BioGPT research results.

Usage:
    python scripts/generate_charts.py

Outputs PNG files to assets/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BIO_COLOR = "#2563eb"
PYTHIA_COLOR = "#dc2626"
ACCENT = "#16a34a"
GRAY = "#6b7280"


def pruning_curve():
    """BioGPT vs Pythia-160M pruning resilience."""

    bio_sparsity = [0, 25, 45, 70, 90]
    bio_ppl      = [17.5, 18.5, 21.2, 61.7, 1084]
    bio_params   = [294, 231, 180, 116, 65]

    pyth_sparsity = [0, 25, 50, 70, 90]
    pyth_ppl      = [31.5, 37.7, 41.4, 106.8, 889]
    pyth_params   = [162, 141, 120, 103, 86]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(bio_sparsity, bio_ppl, "o-", color=BIO_COLOR, linewidth=2.5,
                markersize=8, label="BioGPT (294M, dendritic)", zorder=5)
    ax.semilogy(pyth_sparsity, pyth_ppl, "s--", color=PYTHIA_COLOR, linewidth=2.5,
                markersize=8, label="Pythia-160M (standard transformer)", zorder=5)

    # Highlight the iso-parameter comparison
    ax.annotate(
        f"BioGPT @ 45% prune\n180M params, ppl={bio_ppl[2]}",
        xy=(45, 21.2), xytext=(55, 14),
        fontsize=10, color=BIO_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BIO_COLOR, lw=1.5),
    )
    ax.annotate(
        f"Pythia unpruned\n162M params, ppl={pyth_ppl[0]}",
        xy=(0, 31.5), xytext=(12, 55),
        fontsize=10, color=PYTHIA_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PYTHIA_COLOR, lw=1.5),
    )

    # Shade the "BioGPT wins" region
    ax.axhspan(0, 31.5, xmin=0, xmax=0.5, alpha=0.06, color=BIO_COLOR)

    ax.set_xlabel("Weight Sparsity (%)")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Pruning Resilience: BioGPT vs Pythia-160M")
    ax.set_xticks([0, 25, 45, 50, 70, 90])
    ax.set_xlim(-3, 95)
    ax.set_ylim(10, 2000)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    fig.savefig(os.path.join(ASSETS, "pruning_curve.png"))
    plt.close(fig)
    print("  Saved pruning_curve.png")


def ablation_chart():
    """Ablation study: impact of each biological feature."""

    configs = [
        "(a) Full\nBranch",
        "(b) No\nFFN",
        "(c) No\nNMDA",
        "(d) NMDA\nOnly",
        "(e) Bare\n(=MHA)",
        "(f) Std\nGPT",
    ]
    val_losses = [1.422, 1.387, 1.394, 1.380, 1.378, 1.569]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = [BIO_COLOR] * 5 + [PYTHIA_COLOR]
    bars = ax.bar(configs, val_losses, color=colors, width=0.6, edgecolor="white",
                  linewidth=1.5, zorder=3)

    for bar, v in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation Study: Contribution of Biological Features")
    ax.set_ylim(1.30, 1.62)
    ax.grid(True, alpha=0.3, axis="y")

    # Add a horizontal line for std GPT baseline
    ax.axhline(y=1.569, color=PYTHIA_COLOR, linestyle=":", alpha=0.5, linewidth=1.5)
    ax.text(5.5, 1.575, "Standard GPT baseline", ha="center", va="bottom",
            fontsize=9, color=PYTHIA_COLOR, fontstyle="italic")

    fig.savefig(os.path.join(ASSETS, "ablation_chart.png"))
    plt.close(fig)
    print("  Saved ablation_chart.png")


def cortical_layout():
    """Visualize the cortical layout: neurons and branches per layer."""

    layers = list(range(12))
    zones = ["EARLY"] * 4 + ["MID"] * 4 + ["LATE"] * 4

    neuron_configs = [
        [8, 8], [8, 8], [8, 8], [8, 8],        # early
        [8, 6, 4], [8, 6, 4], [8, 6, 4], [8, 6, 4],  # mid
        [4, 4], [4, 4], [4, 4], [4, 4],          # late
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: branches per layer (stacked bars for each neuron)
    zone_colors = {"EARLY": "#3b82f6", "MID": "#8b5cf6", "LATE": "#f59e0b"}
    neuron_alphas = [1.0, 0.7, 0.5]

    for i, (nc, zone) in enumerate(zip(neuron_configs, zones)):
        bottom = 0
        for j, nb in enumerate(nc):
            ax1.bar(i, nb, bottom=bottom, color=zone_colors[zone],
                    alpha=neuron_alphas[j], edgecolor="white", linewidth=1, width=0.7)
            ax1.text(i, bottom + nb / 2, str(nb), ha="center", va="center",
                     fontsize=8, fontweight="bold", color="white")
            bottom += nb

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Total Branches")
    ax1.set_title("Branches per Layer")
    ax1.set_xticks(layers)

    # Zone labels
    for zone, start, end, color in [("Early (sensory)", 0, 3, "#3b82f6"),
                                     ("Middle (association)", 4, 7, "#8b5cf6"),
                                     ("Late (abstract)", 8, 11, "#f59e0b")]:
        mid = (start + end) / 2
        ax1.text(mid, -2.5, zone, ha="center", fontsize=9, color=color, fontweight="bold")

    ax1.set_ylim(0, 22)

    # Right: branch size (d_branch) per layer
    branch_sizes = []
    for nc in neuron_configs:
        sizes = [768 // nb for nb in nc]
        branch_sizes.append(sizes)

    for i, (sizes, zone) in enumerate(zip(branch_sizes, zones)):
        for j, s in enumerate(sizes):
            marker = ["o", "D", "^"][j]
            ax2.scatter(i, s, color=zone_colors[zone], s=80,
                        marker=marker, alpha=neuron_alphas[j], zorder=5,
                        edgecolors="white", linewidths=0.5)

    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Branch Size (d_branch)")
    ax2.set_title("Branch Granularity")
    ax2.set_xticks(layers)
    ax2.set_ylim(60, 220)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cortical Layout: Biologically-Inspired Layer Structure", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "cortical_layout.png"))
    plt.close(fig)
    print("  Saved cortical_layout.png")


def param_comparison():
    """Parameter count comparison at pruning levels."""

    bio_sparsity = [0, 25, 45, 70, 90]
    bio_params   = [294, 231, 180, 116, 65]
    bio_loss     = [2.86, 2.92, 3.05, 4.12, 6.99]

    pyth_sparsity = [0, 25, 50, 70, 90]
    pyth_params   = [162, 141, 120, 103, 86]
    pyth_loss     = [3.45, 3.63, 3.72, 4.67, 6.79]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bio_params, bio_loss, "o-", color=BIO_COLOR, linewidth=2.5,
            markersize=9, label="BioGPT (dendritic)", zorder=5)
    ax.plot(pyth_params, pyth_loss, "s--", color=PYTHIA_COLOR, linewidth=2.5,
            markersize=9, label="Pythia-160M (standard)", zorder=5)

    for sp, p, l in zip(bio_sparsity, bio_params, bio_loss):
        if l < 7:
            ax.annotate(f"{sp}%", (p, l), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, color=BIO_COLOR)
    for sp, p, l in zip(pyth_sparsity, pyth_params, pyth_loss):
        if l < 7:
            ax.annotate(f"{sp}%", (p, l), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, color=PYTHIA_COLOR)

    ax.set_xlabel("Non-zero Parameters (M)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Efficiency Frontier: Loss vs Active Parameters")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(2.5, 7.5)

    fig.savefig(os.path.join(ASSETS, "efficiency_frontier.png"))
    plt.close(fig)
    print("  Saved efficiency_frontier.png")


if __name__ == "__main__":
    print("Generating charts...")
    pruning_curve()
    ablation_chart()
    cortical_layout()
    param_comparison()
    print("Done! Charts saved to assets/")
