import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

C_STD = "#3A86FF"    # Standard Degree+Jaccard
C_ADP = "#8338EC"    # Adaptive (purple — distinct from both)
C_LV  = "#FF6B6B"    # Louvain

def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {name}")

#1. Node and edge reduction: Standard vs Adaptive 

def plot_adaptive_compression(results_std: dict, results_adp: dict):
    """
    results_std / results_adp: {dataset_name: eval_dict}
    """
    datasets  = list(results_std.keys())
    std_node  = [results_std[d]["node_reduction_%"] for d in datasets]
    adp_node  = [results_adp[d]["node_reduction_%"] for d in datasets]
    std_edge  = [results_std[d]["edge_reduction_%"] for d in datasets]
    adp_edge  = [results_adp[d]["edge_reduction_%"] for d in datasets]

    x = np.arange(len(datasets))
    w = 0.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Standard Degree+Jaccard vs Adaptive Degree-Weighted Jaccard\n"
                 "(Adaptive threshold θ computed automatically from graph structure)",
                 fontsize=12, fontweight="bold")

    for ax, s_vals, a_vals, ylabel, title in [
        (ax1, std_node, adp_node, "Node Reduction (%)", "Node Reduction"),
        (ax2, std_edge, adp_edge, "Edge Reduction (%)", "Edge Reduction"),
    ]:
        b1 = ax.bar(x - w/2, s_vals, w, label="Standard (manual θ)",
                    color=C_STD, alpha=0.88, edgecolor="white")
        b2 = ax.bar(x + w/2, a_vals, w, label="Adaptive (auto θ)",
                    color=C_ADP, alpha=0.88, edgecolor="white")
        for bars, vals in [(b1, s_vals), (b2, a_vals)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.8,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(max(s_vals), max(a_vals)) * 1.22 + 5)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    _save(fig, "adaptive_compression_bar.png")

#  2. Hub preservation check 

def plot_hub_preservation(hub_data: dict):
    """
    hub_data = {
      "Karate": {
        "hub_degrees": [17,16,12],
        "std_supernode_sizes": [3,2,1],
        "adp_supernode_sizes": [1,1,1],
      }, ...
    }
    Shows that in the adaptive method, top hubs stay as singletons.
    """
    datasets = [d for d in hub_data if d != "Petersen"]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]
    fig.suptitle("Hub Node Preservation: Standard vs Adaptive\n"
                 "Supernode size = 1 means hub was NOT merged (preserved)",
                 fontsize=12, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        d   = hub_data[ds]
        hubs   = [f"Hub {i+1}\n(deg {deg})"
                  for i, deg in enumerate(d["hub_degrees"])]
        std_sz = d["std_supernode_sizes"]
        adp_sz = d["adp_supernode_sizes"]
        x = np.arange(len(hubs))
        w = 0.3
        ax.bar(x - w/2, std_sz, w, label="Standard", color=C_STD, alpha=0.88)
        ax.bar(x + w/2, adp_sz, w, label="Adaptive", color=C_ADP, alpha=0.88)
        ax.axhline(1, color="gray", ls="--", lw=1.2, label="Singleton (preserved)")
        ax.set_xticks(x); ax.set_xticklabels(hubs, fontsize=9)
        ax.set_ylabel("Supernode Size (nodes merged into hub's group)")
        ax.set_title(ds); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(std_sz), 2) * 1.3)

    _save(fig, "adaptive_hub_preservation.png")

# 3. Auto-computed θ vs manual θ 

def plot_threshold_comparison(threshold_data: dict):
    """
    threshold_data = {"Petersen": {"manual": 0.2, "adaptive": 0.15}, ...}
    """
    datasets = list(threshold_data.keys())
    manual   = [threshold_data[d]["manual"]   for d in datasets]
    adaptive = [threshold_data[d]["adaptive"] for d in datasets]

    x = np.arange(len(datasets))
    w = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, manual,   w, label="Manual θ (hand-tuned)",
                color=C_STD, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + w/2, adaptive, w, label="Adaptive θ (auto-computed)",
                color=C_ADP, alpha=0.88, edgecolor="white")

    for bars, vals in [(b1, manual), (b2, adaptive)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Similarity Threshold θ")
    ax.set_ylim(0, max(max(manual), max(adaptive)) * 1.25)
    ax.set_title("Manual (Hand-Tuned) vs Auto-Computed Adaptive Threshold θ\n"
                 "Adaptive θ = max(θ_min,  μ_J − α · σ_J)  where μ_J, σ_J are sampled per graph",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    _save(fig, "adaptive_threshold_auto.png")

#  4. Three-way comparison: Standard vs Adaptive vs Louvain 

def plot_three_way(results_std: dict, results_adp: dict, results_lv: dict):
    
    datasets = list(results_std.keys())
    std_vals = [results_std[d]["node_reduction_%"] for d in datasets]
    adp_vals = [results_adp[d]["node_reduction_%"] for d in datasets]
    lv_vals  = [results_lv[d]["node_reduction_%"]  for d in datasets]

    x = np.arange(len(datasets))
    w = 0.22
    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - w,   std_vals, w, label="Standard Deg+Jaccard",
                color=C_STD, alpha=0.88, edgecolor="white")
    b2 = ax.bar(x,       adp_vals, w, label="Adaptive Deg+DWJ (proposed)",
                color=C_ADP, alpha=0.88, edgecolor="white")
    b3 = ax.bar(x + w,   lv_vals,  w, label="Louvain (baseline)",
                color=C_LV,  alpha=0.88, edgecolor="white")

    for bars, vals in [(b1, std_vals), (b2, adp_vals), (b3, lv_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Node Reduction (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Three-Way Comparison: Standard vs Adaptive vs Louvain\n"
                 "Adaptive method bridges the compression gap without manual threshold tuning",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    _save(fig, "adaptive_three_way.png")
