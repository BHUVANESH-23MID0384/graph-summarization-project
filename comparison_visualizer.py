import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

C_DJ  = "#3A86FF"
C_LV  = "#FF6B6B"
ALPHA = 0.88


def _save(fig, name: str):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {name}")


def _safe_float(v, fallback: float = 0.0) -> float:
    """Convert None/inf to a safe fallback for plotting."""
    if v is None:
        return fallback
    f = float(v)
    return fallback if (f != f or f == float('inf') or f == float('-inf')) else f


# 1. Comparison table 

def plot_comparison_table(results: dict):
    datasets = list(results.keys())
    cols = ["Dataset", "Method", "Nodes→", "Edges→",
            "Node Red.%", "Edge Red.%", "Modularity Q",
            "Runtime (s)", "Connected?"]

    rows = []
    for ds in datasets:
        for method_key, label in [("dj", "Deg+Jaccard"), ("lv", "Louvain")]:
            m = results[ds][method_key]
            t = results[ds][f"{method_key}_time"]
            # FIX: show mean±std for Louvain Q when available
            if method_key == "lv" and m.get("modularity_Q_mean") is not None:
                q_str = f"{m['modularity_Q_mean']:.4f}±{m.get('modularity_Q_std', 0):.4f}"
            elif m["modularity_Q"] is not None:
                q_str = f"{m['modularity_Q']}"
            else:
                q_str = "—"
            rows.append([
                ds if method_key == "dj" else "",
                label,
                f"{m['original_nodes']}→{m['summary_nodes']}",
                f"{m['original_edges']}→{m['summary_edges']}",
                f"{m['node_reduction_%']:.1f}%",
                f"{m['edge_reduction_%']:.1f}%",
                q_str,
                f"{t:.4f}",
                "✓" if m['summary_connected'] else "✗",
            ])

    fig, ax = plt.subplots(figsize=(16, len(rows) * 0.65 + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows):
        bg = "#EAF4FF" if row[1] == "Deg+Jaccard" else "#FFF0F0"
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)
    ax.set_title(
        "Head-to-Head: Degree+Jaccard vs Louvain  (Louvain Q = mean±std over 10 seeds)",
        fontsize=12, fontweight="bold", pad=14)
    _save(fig, "comparison_table.png")


#2. Node reduction bar 

def plot_node_reduction_bar(results: dict):
    datasets = list(results.keys())
    dj_vals  = [_safe_float(results[d]["dj"]["node_reduction_%"]) for d in datasets]
    lv_vals  = [_safe_float(results[d]["lv"]["node_reduction_%"]) for d in datasets]

    x, w = np.arange(len(datasets)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, dj_vals, w, label="Degree+Jaccard", color=C_DJ, alpha=ALPHA, edgecolor="white")
    b2 = ax.bar(x + w/2, lv_vals, w, label="Louvain",        color=C_LV, alpha=ALPHA, edgecolor="white")
    for bars, vals in [(b1, dj_vals), (b2, lv_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Node Reduction (%)"); ax.set_ylim(0, max(max(dj_vals), max(lv_vals), 10) * 1.22)
    ax.set_title("Node Reduction %: Degree+Jaccard vs Louvain", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "node_reduction_bar.png")


#  3. Modularity bar 

def plot_modularity_bar(results: dict):
    datasets = list(results.keys())
    # FIX: use mean Q for Louvain where available
    dj_vals = [_safe_float(results[d]["dj"]["modularity_Q"]) for d in datasets]
    lv_vals = [_safe_float(results[d]["lv"].get("modularity_Q_mean") or
                           results[d]["lv"]["modularity_Q"]) for d in datasets]
    lv_errs = [_safe_float(results[d]["lv"].get("modularity_Q_std", 0)) for d in datasets]

    x, w = np.arange(len(datasets)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, dj_vals, w, label="Degree+Jaccard", color=C_DJ, alpha=ALPHA, edgecolor="white")
    b2 = ax.bar(x + w/2, lv_vals, w, label="Louvain (mean±std)", color=C_LV, alpha=ALPHA, edgecolor="white",
                yerr=lv_errs, capsize=4, error_kw={"ecolor": "#c0392b", "lw": 1.5})
    for bars, vals in [(b1, dj_vals), (b2, lv_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    # FIX: Q=0 reference line so negative values are visible
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Modularity Q")
    ymin = min(min(dj_vals), min(lv_vals)) - 0.05
    # FIX: ensure ymax is always above zero so Q=0 reference line is visible
    ymax = max(max(dj_vals), max(lv_vals), 0.05) * 1.25
    ax.set_ylim(min(ymin, -0.05), ymax)
    ax.set_title("Modularity Q: Degree+Jaccard vs Louvain\n"
                 "Note: negative Q is expected for structural-equivalence partitions",
                 fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "modularity_bar.png")


#  4. Runtime bar 

def plot_runtime_bar(results: dict):
    datasets = list(results.keys())
    dj_vals  = [results[d]["dj_time"] for d in datasets]
    lv_vals  = [results[d]["lv_time"] for d in datasets]

    x, w = np.arange(len(datasets)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, dj_vals, w, label="Degree+Jaccard", color=C_DJ, alpha=ALPHA, edgecolor="white")
    b2 = ax.bar(x + w/2, lv_vals, w, label="Louvain",        color=C_LV, alpha=ALPHA, edgecolor="white")
    for bars, vals in [(b1, dj_vals), (b2, lv_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                    f"{val:.4f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Comparison: Degree+Jaccard vs Louvain\n"
                 "(Louvain runtime includes 10-seed averaging)",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "runtime_bar.png")


#  5. Radar chart 

def plot_radar(results: dict):
    
    # Only include datasets where both methods produce meaningful compression
    use = [d for d in results
           if results[d]["lv"]["node_reduction_%"] > 0
           and results[d]["dj"]["edge_reduction_%"] < 100.0]  
    if not use:
        use = list(results.keys())[:2]   

    categories = ["Node Red.", "Edge Red.", "Modularity Q", "Speed (inv.)", "Connectivity"]

   
    max_node = max(max(_safe_float(results[d]["dj"]["node_reduction_%"]),
                       _safe_float(results[d]["lv"]["node_reduction_%"])) for d in use) or 1.0
    max_edge = max(max(_safe_float(results[d]["dj"]["edge_reduction_%"]),
                       _safe_float(results[d]["lv"]["edge_reduction_%"])) for d in use) or 1.0
    # For Q, only use positive values in the max to avoid div-by-near-zero
    q_vals = [_safe_float(results[d]["lv"].get("modularity_Q_mean") or
                          results[d]["lv"]["modularity_Q"]) for d in use]
    max_mod = max(q_vals) if max(q_vals) > 0 else 1.0
    max_time = max(max(results[d]["dj_time"], results[d]["lv_time"]) for d in use) or 1.0

    def avg(vals): return float(np.mean(vals)) if vals else 0.0

    def scores(mk):
        node_s = avg([_safe_float(results[d][mk]["node_reduction_%"]) for d in use]) / max_node
        edge_s = avg([_safe_float(results[d][mk]["edge_reduction_%"]) for d in use]) / max_edge
        if mk == "lv":
            q_raw = avg([_safe_float(results[d][mk].get("modularity_Q_mean") or
                                     results[d][mk]["modularity_Q"]) for d in use])
        else:
            q_raw = avg([_safe_float(results[d][mk]["modularity_Q"]) for d in use])
        # FIX: clamp Q score to [0,1] — negative Q maps to 0
        mod_s = max(0.0, q_raw / max_mod)
        spd_s = 1.0 - avg([results[d][f"{mk}_time"] for d in use]) / max_time
        con_s = avg([float(results[d][mk]["summary_connected"]) for d in use])
        return [node_s, edge_s, mod_s, spd_s, con_s]

    dj_scores = scores("dj")
    lv_scores = scores("lv")

    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    polar_ax: PolarAxes = ax  # type: ignore[assignment]

    for vals, color, label in [(dj_scores, C_DJ, "Degree+Jaccard"),
                                (lv_scores, C_LV, "Louvain")]:
        v = vals + vals[:1]
        polar_ax.plot(angles, v, color=color, lw=2, label=label)
        polar_ax.fill(angles, v, color=color, alpha=0.15)

    polar_ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    polar_ax.set_ylim(0, 1)
    polar_ax.set_title(
        "Method Trade-off Profile\n"
        f"(averaged over: {', '.join(use)})\n"
        "Edge Red. uses % (bounded); Q score clamped ≥ 0",
        fontsize=10, fontweight="bold", pad=20)
    polar_ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    polar_ax.grid(alpha=0.3)
    _save(fig, "radar_chart.png")