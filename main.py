import sys, os, tracemalloc, time
from typing import Optional
sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx
from pathlib import Path

from graph_summarizer        import GraphSummarizer
from louvain_comparator      import LouvainComparator
from adaptive_summarizer     import AdaptiveGraphSummarizer
from hierarchical_summarizer import HierarchicalSummarizer

from dataset_loader import (load_petersen, load_karate_club,
                             load_erdos_renyi, load_barabasi_albert)
from visualizer import (plot_original_and_summary, plot_degree_distribution,
                        plot_compression_metrics, plot_threshold_sensitivity)
from comparison_visualizer import (plot_comparison_table, plot_node_reduction_bar,
                                   plot_modularity_bar, plot_runtime_bar, plot_radar)
from adaptive_visualizer import (plot_adaptive_compression, plot_hub_preservation,
                                  plot_threshold_comparison, plot_three_way)
from hierarchical_visualizer import (plot_compression_curve, plot_modularity_trajectory,
                                      plot_node_lineage, plot_all_datasets)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SEP = "─" * 60


# Shared helper 

def print_metrics(m: dict, perf: Optional[dict] = None, label: str = ""):
    print(f"\n  [{label}]")
    print(f"    Nodes  : {m['original_nodes']} → {m['summary_nodes']}")
    print(f"    Edges  : {m['original_edges']} → {m['summary_edges']}")
    print(f"    Node%  : {m['node_reduction_%']:.2f}%")
    print(f"    Edge%  : {m['edge_reduction_%']:.2f}%")
    # FIX: show mean±std for Louvain Q, plain Q for others
    if m.get("modularity_Q_mean") is not None:
        print(f"    Mod Q  : {m['modularity_Q_mean']:.4f} ± {m.get('modularity_Q_std',0):.4f} (10 seeds)")
    else:
        print(f"    Mod Q  : {m.get('modularity_Q', '—')}")
    print(f"    Conn.  : {m['summary_connected']}")
    if perf:
        print(f"    Time   : {perf['time_seconds']:.4f}s")


def _run_timed(algo, G, **kwargs):
    """Run algo.summarize(G) with tracemalloc timing. Returns (metrics, time_s)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    algo.summarize(G, **kwargs)
    t1 = time.perf_counter()
    _, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return algo.evaluate(), round(t1 - t0, 4)


# Exp 1: Petersen 

def experiment_petersen():
    print(f"\n{'▓'*55}\n  EXP 1 — Petersen Graph\n{'▓'*55}")
    G  = load_petersen()
    gs = GraphSummarizer(theta=0.2, epsilon=1)
    pf = gs.timing_and_memory(G)
    m  = gs.evaluate()
    print_metrics(m, pf, "Petersen — Degree+Jaccard")
    plot_original_and_summary(G, gs.summarized_graph, gs.supernode_members,
        save_path=str(OUTPUT_DIR/"petersen_comparison.png"))
    plot_degree_distribution(G, gs.summarized_graph,
        save_path=str(OUTPUT_DIR/"petersen_degree_dist.png"))
    plot_compression_metrics(m, save_path=str(OUTPUT_DIR/"petersen_metrics.png"))
    return m, pf


# Exp 2: Karate Club 

def experiment_karate():
    print(f"\n{'▓'*55}\n  EXP 2 — Karate Club\n{'▓'*55}")
    G  = load_karate_club()
    gs = GraphSummarizer(theta=0.3, epsilon=2)
    pf = gs.timing_and_memory(G)
    m  = gs.evaluate()
    print_metrics(m, pf, "Karate — Degree+Jaccard")
    plot_original_and_summary(G, gs.summarized_graph, gs.supernode_members,
        save_path=str(OUTPUT_DIR/"karate_comparison.png"))
    plot_degree_distribution(G, gs.summarized_graph,
        save_path=str(OUTPUT_DIR/"karate_degree_dist.png"))
    plot_compression_metrics(m, save_path=str(OUTPUT_DIR/"karate_metrics.png"))
    return m, pf


# Exp 3: Erdős–Rényi 

def experiment_erdos_renyi():
    print(f"\n{'▓'*55}\n  EXP 3 — Erdős–Rényi G(200, 0.04)\n{'▓'*55}")
    G   = load_erdos_renyi(n=200, p=0.04)
    lcc = max(nx.connected_components(G), key=len)
    G   = G.subgraph(lcc).copy()
    gs  = GraphSummarizer(theta=0.3, epsilon=1)
    pf  = gs.timing_and_memory(G)
    m   = gs.evaluate()
    print_metrics(m, pf, "Erdős–Rényi — Degree+Jaccard")
    plot_original_and_summary(G, gs.summarized_graph, gs.supernode_members,
        save_path=str(OUTPUT_DIR/"er_comparison.png"))
    plot_degree_distribution(G, gs.summarized_graph,
        save_path=str(OUTPUT_DIR/"er_degree_dist.png"))
    plot_compression_metrics(m, save_path=str(OUTPUT_DIR/"er_metrics.png"))
    return m, pf


# Exp 4: Barabasi–Albert 

def experiment_barabasi():
    print(f"\n{'▓'*55}\n  EXP 4 — Barabási–Albert G(300, m=2)\n{'▓'*55}")
    G  = load_barabasi_albert(n=300, m=2)
    gs = GraphSummarizer(theta=0.25, epsilon=2)
    pf = gs.timing_and_memory(G)
    m  = gs.evaluate()
    print_metrics(m, pf, "Barabási–Albert — Degree+Jaccard")
    plot_original_and_summary(G, gs.summarized_graph, gs.supernode_members,
        save_path=str(OUTPUT_DIR/"ba_comparison.png"))
    plot_degree_distribution(G, gs.summarized_graph,
        save_path=str(OUTPUT_DIR/"ba_degree_dist.png"))
    plot_compression_metrics(m, save_path=str(OUTPUT_DIR/"ba_metrics.png"))
    return m, pf


#Exp 5: Threshold Sensitivity 

def experiment_threshold_sensitivity():
    print(f"\n{'▓'*55}\n  EXP 5 — Threshold Sensitivity\n{'▓'*55}")
    G = load_karate_club()
    plot_threshold_sensitivity(
        G,
        theta_values=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        epsilon=2,
        save_path=str(OUTPUT_DIR/"threshold_sensitivity.png"))
    print("  Saved threshold_sensitivity.png")


# Exp 6: Louvain Comparison 

def experiment_louvain_comparison():
    print(f"\n{'▓'*55}")
    print("  EXP 6 — Head-to-Head: Deg+Jaccard vs Louvain (10 seeds)")
    print(f"{'▓'*55}")

    G_pet = load_petersen()
    G_kar = load_karate_club()
    G_er  = load_erdos_renyi(n=200, p=0.04)
    G_er  = G_er.subgraph(max(nx.connected_components(G_er), key=len)).copy()
    G_ba  = load_barabasi_albert(n=300, m=2)

    configs = {
        "Petersen":        (G_pet, 0.2,  1),
        "Karate Club":     (G_kar, 0.3,  2),
        "Erdős–Rényi":     (G_er,  0.3,  1),
        "Barabási–Albert": (G_ba,  0.25, 2),
    }

    results = {}

    for name, (G, theta, eps) in configs.items():
        print(f"\n  {name}  (θ={theta}, ε={eps})")
        print(f"  {SEP}")

        gs = GraphSummarizer(theta=theta, epsilon=eps)
        dj_m, dj_time = _run_timed(gs, G)
        print_metrics(dj_m, {"time_seconds": dj_time}, "Degree+Jaccard")

        # FIX: LouvainComparator now runs 10 seeds internally
        lv = LouvainComparator()
        lv_m, lv_time = _run_timed(lv, G)
        print_metrics(lv_m, {"time_seconds": lv_time}, "Louvain (10-seed avg)")

        results[name] = {
            "dj": dj_m, "lv": lv_m,
            "dj_time": dj_time, "lv_time": lv_time,
        }

    print("\n  Generating comparison plots...")
    plot_comparison_table(results)
    plot_node_reduction_bar(results)
    plot_modularity_bar(results)
    plot_runtime_bar(results)
    plot_radar(results)

    print("\n  Summary (per dataset):")
    for name, r in results.items():
        faster   = "Deg+Jaccard" if r["dj_time"] < r["lv_time"] else "Louvain"
        dj_q     = r["dj"]["modularity_Q"] or 0
        lv_q     = r["lv"].get("modularity_Q_mean") or r["lv"]["modularity_Q"] or 0
        higher_q = "Louvain" if lv_q > dj_q else "Deg+Jaccard"
        print(f"    {name:<22}: Faster={faster}  Higher-Q={higher_q}")

    return results


# Exp 7: Adaptive DWJ 

def experiment_adaptive(louvain_results: Optional[dict] = None):
    print(f"\n{'▓'*55}")
    print("  EXP 7 — Adaptive Deg-Weighted Jaccard vs Standard")
    print(f"{'▓'*55}")

    G_er = load_erdos_renyi(n=200, p=0.04)
    G_er = G_er.subgraph(max(nx.connected_components(G_er), key=len)).copy()

    configs = {
        "Petersen":        (load_petersen(),                  0.20, 1),
        "Karate Club":     (load_karate_club(),               0.30, 2),
        "Barabási–Albert": (load_barabasi_albert(n=300, m=2), 0.25, 2),
        "Erdős–Rényi":     (G_er,                            0.30, 1),
    }

    res_std, res_adp, thresh_data, hub_data = {}, {}, {}, {}

    for name, (G, theta, eps) in configs.items():
        print(f"\n  {name}")

        gs = GraphSummarizer(theta=theta, epsilon=eps)
        gs.summarize(G)
        ms = gs.evaluate()
        res_std[name] = ms
        print_metrics(ms, label=f"{name} — Standard (θ={theta})")

        ad = AdaptiveGraphSummarizer(theta_min=0.15, alpha=0.5, beta=0.4, epsilon=eps)
        ad.summarize(G)
        ma = ad.evaluate()
        res_adp[name] = ma
        print_metrics(ma, label=f"{name} — Adaptive (θ=auto={ma['theta_used']})")

        thresh_data[name] = {"manual": theta, "adaptive": ma["theta_used"]}

        if name != "Petersen":
            degs = dict(G.degree())
            top3 = sorted(degs, key=lambda x: -degs[x])[:3]
            hub_data[name] = {
                "hub_degrees":         [degs[h] for h in top3],
                "std_supernode_sizes": [len(gs.supernode_members.get(
                                            gs.node_to_supernode.get(h, ""), [])) for h in top3],
                "adp_supernode_sizes": [len(ad.supernode_members.get(
                                            ad.node_to_supernode.get(h, ""), [])) for h in top3],
            }

    # FIX: compute Louvain node_reduction live from Exp 6 results if available,
    # otherwise run LouvainComparator fresh — no hardcoded numbers
    if louvain_results is not None:
        res_lv = {name: {"node_reduction_%": louvain_results[name]["lv"]["node_reduction_%"]}
                  for name in configs if name in louvain_results}
    else:
        res_lv = {}
        for name, (G, _, _) in configs.items():
            lv = LouvainComparator()
            lv.summarize(G)
            lm = lv.evaluate()
            res_lv[name] = {"node_reduction_%": lm["node_reduction_%"]}

    print("\n  Generating adaptive comparison plots...")
    plot_adaptive_compression(res_std, res_adp)
    plot_hub_preservation(hub_data)
    plot_threshold_comparison(thresh_data)
    plot_three_way(res_std, res_adp, res_lv)

    return res_std, res_adp


# Exp 8: Hierarchical 

def experiment_hierarchical():
    print(f"\n{'▓'*55}")
    print("  EXP 8 — Hierarchical Multi-Level Summarization")
    print(f"{'▓'*55}")

    G_er = load_erdos_renyi(n=200, p=0.04)
    G_er = G_er.subgraph(max(nx.connected_components(G_er), key=len)).copy()
    G_ba = load_barabasi_albert(n=300, m=2)

    configs = {
        "Petersen":        (load_petersen(),  0.20, 1),
        "Karate Club":     (load_karate_club(), 0.25, 2),
        "Barabási–Albert": (G_ba,             0.25, 2),
        "Erdős–Rényi":     (G_er,             0.30, 1),
    }

    all_hs = {}

    for name, (G, theta, eps) in configs.items():
        print(f"\n  {name}  (θ={theta}, ε={eps})")
        hs = HierarchicalSummarizer(theta=theta, epsilon=eps, max_levels=10)
        hs.build(G)
        hs.print_summary()

        degs = dict(G.degree())
        hub  = max(degs, key=lambda x: degs[x])
        lin  = hs.node_lineage(hub)
        print(f"\n  Lineage of hub node {hub} (deg={degs[hub]}):")
        for entry in lin:
            print(f"    L{entry['level']}: group_size={entry['group_size']}, "
                  f"supernode={entry['supernode']}")

        all_hs[name] = (hs, G)
        plot_compression_curve(hs, name)
        if hs.levels[-1].level_index > 0:
            plot_node_lineage(hs, G, name)

    active = {n: hs for n, (hs, _) in all_hs.items()
              if hs.levels[-1].level_index > 0}
    if active:
        plot_modularity_trajectory(active)
        all_hs_flat = {n: hs for n, (hs, _) in all_hs.items()}
        plot_all_datasets(all_hs_flat)

    return all_hs


# Main 

if __name__ == "__main__":
    print("\n" + "█"*55)
    
    print(" Efficient Graph Summarization Using Structural and Degree-Based Similarity")
    print("█"*55)

    experiment_petersen()
    experiment_karate()
    experiment_erdos_renyi()
    experiment_barabasi()
    experiment_threshold_sensitivity()
    louvain_results = experiment_louvain_comparison()
    experiment_adaptive(louvain_results)    # FIX: pass live Louvain results
    experiment_hierarchical()

    print(f"\n{'█'*55}")
    print(" All 8 experiments complete.")
    print(f"  Outputs → {OUTPUT_DIR.resolve()}/")
    print("█"*55)
