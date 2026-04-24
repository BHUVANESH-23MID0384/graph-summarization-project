import time
import tracemalloc
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from networkx.algorithms.community import modularity as nx_modularity

from graph_summarizer import GraphSummarizer, safe_is_connected, _safe_modularity


@dataclass
class HierarchyLevel:
    level_index:             int
    graph:                   nx.Graph
    summarizer:              Optional[GraphSummarizer]
    node_count:              int
    edge_count:              int
    node_reduction_pct:      float
    edge_reduction_pct:      float
    incremental_compression: float
    modularity_Q:            Optional[float]
    original_members:        dict = field(default_factory=dict)
    # supernode_label → frozenset of ORIGINAL node ids (raw type, no str cast)


class HierarchicalSummarizer:
   
    def __init__(self, theta: float = 0.3, epsilon: int = 2,
                 max_levels: int = 10):
        self.theta      = theta
        self.epsilon    = epsilon
        self.max_levels = max_levels

        self.levels:        list[HierarchyLevel]  = []
        self.original_graph: Optional[nx.Graph]  = None
        self.converged_at:   int                  = 0
        self.total_time_s:   float                = 0.0

    def build(self, G: nx.Graph) -> list[HierarchyLevel]:
        # FIX: guard empty graph
        if G.number_of_nodes() == 0:
            self.original_graph = G
            self.levels = []
            return self.levels

        self.original_graph = G
        V0 = G.number_of_nodes()
        E0 = G.number_of_edges()
        t_start = time.perf_counter()

        # Level 0 — original_members keys are RAW node ids (no str cast)
        
        l0 = HierarchyLevel(
            level_index=0,
            graph=G,
            summarizer=None,
            node_count=V0,
            edge_count=E0,
            node_reduction_pct=0.0,
            edge_reduction_pct=0.0,
            incremental_compression=0.0,
            modularity_Q=None,
            original_members={n: frozenset([n]) for n in G.nodes()},
        )
        self.levels    = [l0]
        current_graph  = G
        prev_members   = l0.original_members   # node_id → frozenset[original ids]

        for level_idx in range(1, self.max_levels + 1):
            prev_n = current_graph.number_of_nodes()

            gs    = GraphSummarizer(theta=self.theta, epsilon=self.epsilon)
            g_new = gs.summarize(current_graph)
            new_n = g_new.number_of_nodes()

            if new_n == prev_n:
                self.converged_at = level_idx
                break

            # Provenance: each new supernode label → union of original node ids
            # gs.supernode_members[label] = list of nodes from current_graph
            # Those nodes ARE keys in prev_members (raw type — no str cast needed)
            orig_members_this: dict = {}
            for sn_label, sn_prev_nodes in gs.supernode_members.items():
                combined = frozenset().union(
                    *(prev_members[pl] for pl in sn_prev_nodes
                      if pl in prev_members)     # FIX: direct key lookup, no str()
                )
                orig_members_this[sn_label] = combined

            # Modularity wrt original graph G
            partition = [set(m) for m in orig_members_this.values() if m]
            Q = _safe_modularity(G, partition)

            node_red = round((1 - new_n / V0) * 100, 2)
            edge_red = round((1 - g_new.number_of_edges() / max(E0, 1)) * 100, 2)
            incr     = round((1 - new_n / prev_n) * 100, 2)

            level = HierarchyLevel(
                level_index=level_idx,
                graph=g_new,
                summarizer=gs,
                node_count=new_n,
                edge_count=g_new.number_of_edges(),
                node_reduction_pct=node_red,
                edge_reduction_pct=edge_red,
                incremental_compression=incr,
                modularity_Q=Q,
                original_members=orig_members_this,
            )
            self.levels.append(level)
            prev_members  = orig_members_this
            current_graph = g_new

            if new_n <= 2:
                self.converged_at = level_idx
                break

        self.total_time_s = round(time.perf_counter() - t_start, 4)
        return self.levels

    def node_lineage(self, original_node) -> list[dict]:
        
        lineage = [{
            "level":      0,
            "supernode":  str(original_node),
            "co_members": [],
            "group_size": 1,
        }]

        current_id = original_node
        for lv in self.levels[1:]:
            gs = lv.summarizer
            assert gs is not None, f"Level {lv.level_index} summarizer is None"
            sn = gs.node_to_supernode.get(current_id)
            if sn is None:
                break
            orig = lv.original_members.get(sn, frozenset())
            co   = sorted([m for m in orig if m != original_node])
            lineage.append({
                "level":      lv.level_index,
                "supernode":  sn,
                "co_members": co,
                "group_size": len(orig),
            })
            current_id = sn

        return lineage

    def metrics_table(self) -> list[dict]:
        rows = []
        for lv in self.levels:
            rows.append({
                "level":                     lv.level_index,
                "nodes":                     lv.node_count,
                "edges":                     lv.edge_count,
                "node_reduction_%":          lv.node_reduction_pct,
                "edge_reduction_%":          lv.edge_reduction_pct,
                "incremental_compression_%": lv.incremental_compression,
                "modularity_Q":              lv.modularity_Q,
                "connected": safe_is_connected(lv.graph),
            })
        return rows

    def print_summary(self):
        print(f"\n  Hierarchy summary (θ={self.theta}, ε={self.epsilon})")
        print(f"  {'Level':<7} {'Nodes':>7} {'Edges':>7} "
              f"{'NodeRed%':>10} {'EdgeRed%':>10} {'IncrComp%':>11} {'Q':>8}")
        print(f"  {'─'*65}")
        for row in self.metrics_table():
            q = f"{row['modularity_Q']:.4f}" if row['modularity_Q'] is not None else "  —"
            print(f"  L{row['level']:<6} {row['nodes']:>7} {row['edges']:>7} "
                  f"{row['node_reduction_%']:>9.1f}% "
                  f"{row['edge_reduction_%']:>9.1f}% "
                  f"{row['incremental_compression_%']:>10.1f}% "
                  f"{q:>8}")
        final = self.levels[-1] if self.levels else None
        final_pct = final.node_reduction_pct if final is not None else 0.0
        print(f"\n  Total levels: {len(self.levels)-1}  |  "
              f"Final compression: {final_pct:.1f}%  |  "
              f"Runtime: {self.total_time_s:.4f}s")