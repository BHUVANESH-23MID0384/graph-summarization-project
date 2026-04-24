import time
import tracemalloc
import numpy as np
import networkx as nx
from typing import Optional
from networkx.algorithms.community import louvain_communities, modularity

from graph_summarizer import safe_is_connected

N_SEEDS = 10
SEEDS   = list(range(N_SEEDS))

class LouvainComparator:
    

    def __init__(self, seed: int = 0):
        # seed param kept for API compatibility but multi-seed is always used
        self.seed               = seed
        self.original_graph:   Optional[nx.Graph] = None
        self.communities:      Optional[list]     = None
        self.summarized_graph: Optional[nx.Graph] = None
        self.node_to_community: dict              = {}
        self.community_members: dict              = {}
        # Multi-seed stats (populated after summarize)
        self.q_mean:  Optional[float] = None
        self.q_std:   Optional[float] = None
        self.q_all:   list            = []

    def summarize(self, G: nx.Graph) -> nx.Graph:
        self.original_graph = G

        
        runs = []
        for s in SEEDS:
            comms = louvain_communities(G, seed=s)
            try:
                q = modularity(G, comms)
            except Exception:
                q = float('-inf')
            runs.append((q, s, comms))

        qs = [r[0] for r in runs if r[0] != float('-inf')]
        self.q_all  = [round(q, 4) for q in qs]
        self.q_mean = round(float(np.mean(qs)), 4)  if qs else None
        self.q_std  = round(float(np.std(qs)),  4)  if qs else None

        # Pick the run whose Q is closest to the median
        median_q = float(np.median(qs)) if qs else 0.0
        best = min(runs, key=lambda r: abs(r[0] - median_q) if r[0] != float('-inf') else 1e9)
        self.communities = best[2]

        # Build node → community mapping from the representative run
        self.node_to_community = {}
        self.community_members = {}
        # Narrow type from Optional[list] to list for Pylance
        assert self.communities is not None
        for idx, comm in enumerate(self.communities):
            label = f"C{idx}"
            self.community_members[label] = list(comm)
            for node in comm:
                self.node_to_community[node] = label

        G_comm = nx.Graph()
        G_comm.add_nodes_from(self.community_members.keys())
        for label, members in self.community_members.items():
            G_comm.nodes[label]['members'] = members
            G_comm.nodes[label]['size']    = len(members)

        for u, v in G.edges():
            cu = self.node_to_community[u]
            cv = self.node_to_community[v]
            if cu != cv and not G_comm.has_edge(cu, cv):
                G_comm.add_edge(cu, cv)

        self.summarized_graph = G_comm
        return G_comm

    def evaluate(self) -> dict:
        assert self.original_graph   is not None, "Call summarize() before evaluate()"
        assert self.summarized_graph is not None, "Call summarize() before evaluate()"
        assert self.communities      is not None, "Call summarize() before evaluate()"

        G  = self.original_graph
        Gp = self.summarized_graph
        V,  E  = G.number_of_nodes(),  G.number_of_edges()
        Vp, Ep = Gp.number_of_nodes(), Gp.number_of_edges()

        
        CR  = round(V  / Vp, 4) if Vp > 0 else None
        ERR = round(E  / Ep, 4) if Ep > 0 else None

        try:
            Q = round(modularity(G, self.communities), 4)
        except Exception:
            Q = None

        return {
            "method":               "Louvain",
            "original_nodes":       V,
            "original_edges":       E,
            "summary_nodes":        Vp,
            "summary_edges":        Ep,
            "compression_ratio":    CR,
            "edge_reduction_ratio": ERR,
            "node_reduction_%":     round((1 - Vp / V) * 100, 2) if V > 0 else 0.0,
            "edge_reduction_%":     round((1 - Ep / E) * 100, 2) if E > 0 else 0.0,
            "modularity_Q":         Q,
            "modularity_Q_mean":    self.q_mean,   # FIX: multi-seed mean
            "modularity_Q_std":     self.q_std,    # FIX: multi-seed std
            "original_connected":   safe_is_connected(G),
            "summary_connected":    safe_is_connected(Gp),
            # FIX: expose theta/epsilon keys with None so downstream
            # code using .get("theta") does not KeyError
            "theta":                None,
            "epsilon":              None,
        }

    def timing_and_memory(self, G: nx.Graph) -> dict:
        tracemalloc.start()
        t0 = time.perf_counter()
        self.summarize(G)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "time_seconds":   round(t1 - t0, 4),
            "peak_memory_KB": round(peak / 1024, 2),
        }

