import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

C_NODE = "#3A86FF"
C_EDGE = "#FF6B6B"
C_MOD  = "#8338EC"
C_HUB  = "#FF6B6B"
C_PERI = "#3A86FF"

def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {name}")

# 1. Compression curve per level 
def plot_compression_curve(hs, dataset_name: str):
    rows    = hs.metrics_table()
    levels  = [r["level"] for r in rows]
    node_r  = [r["node_reduction_%"] for r in rows]
    edge_r  = [r["edge_reduction_%"] for r in rows]
    incr    = [r["incremental_compression_%"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Hierarchical Summarization — {dataset_name}\n"
        f"Cumulative compression across {len(levels)-1} levels "
        f"(θ={hs.theta}, ε={hs.epsilon})",
        fontsize=12, fontweight="bold"
    )

    # Cumulative reduction
    ax1.plot(levels, node_r, "o-", color=C_NODE, lw=2.2,
             label="Node reduction %")
    ax1.plot(levels, edge_r, "s--", color=C_EDGE, lw=2.2,
             label="Edge reduction %")
    for lv, nr, er in zip(levels, node_r, edge_r):
        if lv > 0:
            ax1.annotate(f"{nr:.0f}%", (lv, nr),
                         textcoords="offset points", xytext=(4, 5),
                         fontsize=8, color=C_NODE)
    ax1.set_xlabel("Hierarchy Level")
    ax1.set_ylabel("Cumulative Reduction vs Original (%)")
    ax1.set_title("Cumulative Compression")
    ax1.set_ylim(-5, 110)
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xticks(levels)

    # Incremental compression per level
    if len(levels) > 1:
        ax2.bar(levels[1:], incr[1:], color=C_NODE, alpha=0.85,
                edgecolor="white", label="Incremental compression")
        for lv, ic in zip(levels[1:], incr[1:]):
            ax2.text(lv, ic + 0.8, f"{ic:.1f}%",
                     ha="center", va="bottom", fontsize=8.5,
                     fontweight="bold")
    ax2.set_xlabel("Hierarchy Level")
    ax2.set_ylabel("Reduction vs Previous Level (%)")
    ax2.set_title("Per-Level (Incremental) Compression")
    ax2.set_xticks(levels[1:] if len(levels) > 1 else [1])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe = dataset_name.replace(" ", "_").replace("–", "").replace("(", "").replace(")", "")
    _save(fig, f"hierarchy_compression_{safe}.png")

#  2. Modularity Q trajectory 

def plot_modularity_trajectory(results: dict):
    """
    results = {"Karate Club": HierarchicalSummarizer, "BA-300": ...}
    Plots Q vs level for each dataset on the same axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [C_NODE, C_EDGE, C_MOD, "#06D6A0"]
    for (name, hs), color in zip(results.items(), colors):
        rows = hs.metrics_table()
        lvs  = [r["level"] for r in rows if r["modularity_Q"] is not None]
        qs   = [r["modularity_Q"] for r in rows if r["modularity_Q"] is not None]
        if lvs:
            ax.plot(lvs, qs, "o-", color=color, lw=2, label=name)
            ax.annotate(f"L{lvs[-1]}: {qs[-1]:.3f}",
                        (lvs[-1], qs[-1]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=8, color=color)

    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6,
               label="Q = 0 (random baseline)")
    ax.set_xlabel("Hierarchy Level")
    ax.set_ylabel("Modularity Q (wrt original graph)")
    ax.set_title(
        "Modularity Q Trajectory Across Hierarchy Levels\n"
        "Positive Q at deeper levels = hierarchy reveals genuine community structure",
        fontsize=11, fontweight="bold"
    )
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "hierarchy_modularity_trajectory.png")

#  3. Node lineage: hub vs peripheral 

def plot_node_lineage(hs, G, dataset_name: str):
    """
    Plots group_size vs level for the highest-degree node (hub)
    and a randomly chosen low-degree node (peripheral).
    """
    import networkx as nx
    degs = dict(G.degree())
    hub  = max(degs, key=lambda x: degs[x])
    peri = min(degs, key=lambda x: degs[x])

    hub_lin  = hs.node_lineage(hub)
    peri_lin = hs.node_lineage(peri)

    lvs_h  = [e["level"] for e in hub_lin]
    szs_h  = [e["group_size"] for e in hub_lin]
    lvs_p  = [e["level"] for e in peri_lin]
    szs_p  = [e["group_size"] for e in peri_lin]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lvs_h, szs_h, "o-", color=C_HUB, lw=2.2,
            label=f"Hub node {hub} (deg={degs[hub]})")
    ax.plot(lvs_p, szs_p, "s--", color=C_PERI, lw=2.2,
            label=f"Peripheral node {peri} (deg={degs[peri]})")

    ax.set_xlabel("Hierarchy Level")
    ax.set_ylabel("Group Size (original nodes absorbed)")
    ax.set_title(
        f"Node Lineage: Hub vs Peripheral — {dataset_name}\n"
        "Hub nodes resist absorption; peripheral nodes merge early",
        fontsize=11, fontweight="bold"
    )
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(lvs_h)

    # Annotate key events
    for entry in hub_lin:
        if entry["group_size"] > 1 and entry["level"] > 0:
            if entry["level"] == hub_lin[1]["level"] or \
               entry["group_size"] != hub_lin[entry["level"]-1]["group_size"]:
                ax.annotate(
                    f"Hub absorbs\n{entry['group_size']-1} nodes",
                    (entry["level"], entry["group_size"]),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=8, color=C_HUB,
                    arrowprops=dict(arrowstyle="->", color=C_HUB, lw=0.8)
                )
                break

    plt.tight_layout()
    safe = dataset_name.replace(" ", "_").replace("–","").replace("(","").replace(")","")
    _save(fig, f"hierarchy_lineage_{safe}.png")

#  4. All-dataset level comparison 

def plot_all_datasets(results: dict):
    """
    For each dataset: node count vs level, all on one figure.
    Shows how different graph structures compress at different rates.
    """
    n_ds = len(results)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 5), sharey=False)
    fig.suptitle(
        "Hierarchical Summarization Across All Graph Families\n"
        "Compression speed and depth depend on graph structure",
        fontsize=12, fontweight="bold"
    )
    if n_ds == 1:
        axes = [axes]

    colors = [C_NODE, C_EDGE, C_MOD, "#06D6A0"]
    for ax, (name, hs), color in zip(axes, results.items(), colors):
        rows = hs.metrics_table()
        lvs  = [r["level"] for r in rows]
        nds  = [r["nodes"]  for r in rows]
        eds  = [r["edges"]  for r in rows]

        ax.plot(lvs, nds, "o-", color=color, lw=2.2, label="Nodes")
        ax.plot(lvs, eds, "s--", color=color, lw=1.6, alpha=0.6,
                label="Edges")
        ax.set_xlabel("Hierarchy Level")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\n"
                     f"{nds[0]}→{nds[-1]} nodes in {len(lvs)-1} levels")
        ax.set_xticks(lvs)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Mark convergence
        if len(lvs) > 1:
            ax.annotate(f"Final:\n{nds[-1]} nodes\n({hs.levels[-1].node_reduction_pct:.0f}% red.)",
                        (lvs[-1], nds[-1]),
                        textcoords="offset points", xytext=(4, 10),
                        fontsize=7.5, color=color)

    plt.tight_layout()
    _save(fig, "hierarchy_all_datasets.png")

