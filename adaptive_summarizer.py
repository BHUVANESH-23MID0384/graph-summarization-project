import numpy as np
import networkx as nx
import time
import tracemalloc
from collections import defaultdict
from itertools import combinations
from typing import Optional
from networkx.algorithms.community import modularity as nx_modularity

from graph_summarizer import _safe_is_connected, _safe_modularity

class AdaptiveGraphSummarizer:
   
    def __init__(self, theta_min: float = 0.15, alpha: float = 0.5,
                 beta: float = 0.4, epsilon: int = 1):
        self.theta_min = theta_min
        self.alpha     = alpha
        self.beta      = beta
        self.epsilon   = epsilon

        self.original_graph:    Optional[nx.Graph] = None
        self.summarized_graph:  Optional[nx.Graph] = None
        self.node_to_supernode: dict               = {}
        self.supernode_members: dict               = {}
        self.theta_used:          Optional[float]  = None
        self.sample_dwj_mean: Optional[float]  = None
        self.sample_dwj_std:  Optional[float]  = None

    def _compute_degrees(self, G: nx.Graph) -> dict:
        return dict(G.degree())

    def _build_buckets(self, degrees: dict, bucket_size: int = 1) -> dict:
        buckets: dict = defaultdict(list)
        for node, deg in degrees.items():
            buckets[deg // bucket_size].append(node)
        return buckets

    def _jaccard(self, G: nx.Graph, u, v) -> float:
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        union = len(Nu | Nv)
        if union == 0:
            return 1.0
        return len(Nu & Nv) / union

    def _weighted_jaccard(self, G: nx.Graph, u, v,
                          degrees: dict, deg_max: int) -> float:
        
        sim = self._jaccard(G, u, v)
        if deg_max == 0:
            return sim
        penalty = self.beta * (degrees[u] + degrees[v]) / (2 * deg_max)
        return sim * max(0.0, 1.0 - penalty)   # FIX: clamp >= 0

    def _estimate_threshold(self, G: nx.Graph, buckets: dict,
                             degrees: dict, deg_max: int,
                             sample_cap: int = 400) -> float:
        
        sampled = []
        rng = np.random.default_rng(42)

        for nodes in buckets.values():
            if len(nodes) < 2:
                continue
            pairs = list(combinations(nodes, 2))
            k = min(len(pairs), max(1, sample_cap // max(len(buckets), 1)))
            chosen = rng.choice(len(pairs), size=k, replace=False)
            for idx in chosen:
                u, v = pairs[idx]
                if abs(degrees[u] - degrees[v]) <= self.epsilon:
                    # FIX: sample DWJ not Jaccard
                    sampled.append(self._weighted_jaccard(G, u, v, degrees, deg_max))

        if not sampled:
            self.sample_dwj_mean = 0.0
            self.sample_dwj_std  = 0.0
            return self.theta_min

        mu    = float(np.mean(sampled))
        sigma = float(np.std(sampled))
        theta = max(self.theta_min, mu - self.alpha * sigma)

        self.sample_dwj_mean = round(mu, 4)
        self.sample_dwj_std  = round(sigma, 4)
        return round(theta, 4)

    def _make_uf(self, nodes):
        parent = {n: n for n in nodes}
        rank   = {n: 0 for n in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        return find, union

    def summarize(self, G: nx.Graph, bucket_size: int = 1) -> nx.Graph:
        
        if G.number_of_nodes() == 0:
            self.original_graph   = G
            self.summarized_graph = nx.Graph()
            return self.summarized_graph

        self.original_graph = G
        degrees  = self._compute_degrees(G)
        buckets  = self._build_buckets(degrees, bucket_size)
        deg_max  = max(degrees.values()) if degrees else 1

        theta = self._estimate_threshold(G, buckets, degrees, deg_max)
        self.theta_used = theta

        find, union = self._make_uf(list(G.nodes()))

        for nodes in buckets.values():
            if len(nodes) < 2:
                continue
            for u, v in combinations(nodes, 2):
                if abs(degrees[u] - degrees[v]) > self.epsilon:
                    continue
                if self._weighted_jaccard(G, u, v, degrees, deg_max) >= theta:
                    union(u, v)

        groups: dict = defaultdict(list)
        for node in G.nodes():
            groups[find(node)].append(node)

        self.node_to_supernode = {}
        self.supernode_members = {}
        for idx, (root, members) in enumerate(groups.items()):
            label = f"A{idx}"
            self.supernode_members[label] = members
            for m in members:
                self.node_to_supernode[m] = label

        G_prime = nx.Graph()
        G_prime.add_nodes_from(self.supernode_members.keys())
        for label, members in self.supernode_members.items():
            G_prime.nodes[label]['members'] = members
            G_prime.nodes[label]['size']    = len(members)

        for u, v in G.edges():
            su = self.node_to_supernode[u]
            sv = self.node_to_supernode[v]
            if su != sv and not G_prime.has_edge(su, sv):
                G_prime.add_edge(su, sv)

        self.summarized_graph = G_prime
        return G_prime

    def evaluate(self) -> dict:
        assert self.original_graph   is not None, "Call summarize() before evaluate()"
        assert self.summarized_graph is not None, "Call summarize() before evaluate()"

        G  = self.original_graph
        Gp = self.summarized_graph
        V,  E  = G.number_of_nodes(),  G.number_of_edges()
        Vp, Ep = Gp.number_of_nodes(), Gp.number_of_edges()

       
        CR  = round(V  / Vp, 4) if Vp > 0 else None
        ERR = round(E  / Ep, 4) if Ep > 0 else None

        partition = [set(m) for m in self.supernode_members.values()]
        Q = _safe_modularity(G, partition)

        return {
            "method":               "Adaptive Deg+DWJ",
            "original_nodes":       V,
            "original_edges":       E,
            "summary_nodes":        Vp,
            "summary_edges":        Ep,
            "node_reduction_%":     round((1 - Vp / V) * 100, 2) if V > 0 else 0.0,
            "edge_reduction_%":     round((1 - Ep / E) * 100, 2) if E > 0 else 0.0,
            "compression_ratio":    CR,
            "edge_reduction_ratio": ERR,
            "modularity_Q":         Q,
            "original_connected":   _safe_is_connected(G),
            "summary_connected":    _safe_is_connected(Gp),
            # FIX: expose theta as theta_used for unified key access
            "theta":                self.theta_used,
            "theta_used":           self.theta_used,
            "theta_min":            self.theta_min,
            "alpha":                self.alpha,
            "beta":                 self.beta,
            "epsilon":              self.epsilon,
            "dwj_mean_sampled": self.sample_dwj_mean,
            "dwj_std_sampled":  self.sample_dwj_std,
        }

    def timing_and_memory(self, G: nx.Graph, bucket_size: int = 1) -> dict:
        tracemalloc.start()
        t0 = time.perf_counter()
        self.summarize(G, bucket_size)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "time_seconds":   round(t1 - t0, 4),
            "peak_memory_KB": round(peak / 1024, 2),
        }

