import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def plot_original_and_summary(G_orig, G_summ, supernode_members,
                               title_orig="Original Graph G=(V,E)",
                               title_summ="Summarized Graph G'=(V',E')",
                               save_path=None):
    """
    Side-by-side plot of original graph and summarized graph.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Graph Summarization: Before vs After", fontsize=15, fontweight='bold')

    # ---- Original graph ----
    ax1 = axes[0]
    pos_orig = nx.spring_layout(G_orig, seed=42)
    nx.draw_networkx(G_orig, pos=pos_orig, ax=ax1,
                     node_color='#4FC3F7', node_size=500,
                     edge_color='#888', font_size=9, font_weight='bold',
                     with_labels=True)
    ax1.set_title(f"{title_orig}\n|V|={G_orig.number_of_nodes()}, |E|={G_orig.number_of_edges()}",
                  fontsize=11)
    ax1.axis('off')

    # ---- Summarized graph ----
    ax2 = axes[1]
    pos_summ = nx.spring_layout(G_summ, seed=42)
    node_sizes = [300 + 200 * G_summ.nodes[n].get('size', 1) for n in G_summ.nodes()]
    nx.draw_networkx(G_summ, pos=pos_summ, ax=ax2,
                     node_color='#66BB6A', node_size=node_sizes,
                     edge_color='#444', font_size=8, font_weight='bold',
                     with_labels=True)
    ax2.set_title(f"{title_summ}\n|V'|={G_summ.number_of_nodes()}, |E'|={G_summ.number_of_edges()}",
                  fontsize=11)
    ax2.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close(fig)


def plot_degree_distribution(G_orig, G_summ, save_path=None):
    """
    Compare degree distributions of original vs summarized graph.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Degree Distribution Comparison", fontsize=13, fontweight='bold')

    for ax, G, color, label in [
        (axes[0], G_orig, '#4FC3F7', "Original Graph"),
        (axes[1], G_summ, '#66BB6A', "Summarized Graph"),
    ]:
        degrees = [d for _, d in G.degree()]
        ax.hist(degrees, bins=max(1, len(set(degrees))),
                color=color, edgecolor='black', alpha=0.85)
        ax.set_title(f"{label}\n|V|={G.number_of_nodes()}, Avg deg={np.mean(degrees):.2f}")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close(fig)


def plot_compression_metrics(metrics: dict, save_path=None):
    
    labels = ['Node Reduction (%)', 'Edge Reduction (%)']
    values = [metrics['node_reduction_%'], metrics['edge_reduction_%']]
    colors = ['#42A5F5', '#EF5350']

    
    theta_val   = metrics.get('theta_used', metrics.get('theta', '—'))
    epsilon_val = metrics.get('epsilon', '—')

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', width=0.4)
    ax.set_ylim(0, 110)   # FIX: fixed ceiling avoids ylim(0,0) when values are 0
    ax.set_ylabel("Reduction (%)")
    ax.set_title(f"Compression Metrics  [θ={theta_val}, ε={epsilon_val}]",
                 fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close(fig)


def plot_threshold_sensitivity(G, theta_values, epsilon=1, bucket_size=1, save_path=None):
    """
    Show how compression changes with different theta thresholds.
   
    """
    from graph_summarizer import GraphSummarizer

    node_reds, edge_reds = [], []
    for theta in theta_values:
        gs = GraphSummarizer(theta=theta, epsilon=epsilon)
        gs.summarize(G, bucket_size)
        m = gs.evaluate()
        node_reds.append(m['node_reduction_%'])
        edge_reds.append(m['edge_reduction_%'])   # FIX: always finite (0-100)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(theta_values, node_reds, 'o-',  color='#42A5F5', label='Node Reduction (%)')
    ax.plot(theta_values, edge_reds, 's--', color='#66BB6A', label='Edge Reduction (%)')
    ax.set_xlabel("Similarity Threshold θ")
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Threshold Sensitivity Analysis", fontsize=12)
    ax.set_ylim(0, 110)   # FIX: fixed ylim, always valid
    ax.legend()
    ax.grid(alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close(fig)