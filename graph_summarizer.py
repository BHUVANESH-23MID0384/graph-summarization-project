import networkx as nx
import time
import tracemalloc
from collections import defaultdict
from itertools import combinations
from typing import Optional
from networkx.algorithms.community import modularity as nx_modularity

def _safe_is_connected(G: nx.Graph) -> bool:
    
    n = G.number_of_nodes()
    if n <= 1:
        return True
    return nx.is_connected(G)

# Public aliases — imported by louvain_comparator and adaptive_summarizer
safe_is_connected = _safe_is_connected

def safe_err(E: int, Ep: int) -> Optional[float]:
    
    if Ep == 0:
        return None
    return round(E / Ep, 4)

def safe_edge_reduction_pct(E: int, Ep: int) -> float:
   
    if E == 0:
        return 0.0
    return round((1 - Ep / E) * 100, 2)

def safe_modularity(G: nx.Graph, partition: list) -> Optional[float]:
    
    return _safe_modularity(G, partition)

def _safe_modularity(G: nx.Graph, partition: list) -> Optional[float]:
    
    if not partition:
        return None
    # All groups must be non-empty and cover exactly the nodes of G
    covered = set()
    for grp in partition:
        if not grp:
            return None
        covered |= grp
    if covered != set(G.nodes()):
        return None
    try:
        return round(nx_modularity(G, partition), 4)
    except Exception:
        return None

class GraphSummarizer:
    

    def __init__(self, theta: float = 0.3, epsilon: int = 1):
        self.theta   = theta
        self.epsilon = epsilon
        self.original_graph:   Optional[nx.Graph] = None
        self.summarized_graph: Optional[nx.Graph] = None
        self.node_to_supernode: dict = {}
        self.supernode_members: dict = {}

    def _compute_degrees(self, G: nx.Graph) -> dict:
        return dict(G.degree())

    def _build_degree_buckets(self, degrees: dict, bucket_size: int = 1) -> dict:
        buckets: dict = defaultdict(list)
        for node, deg in degrees.items():
            buckets[deg // bucket_size].append(node)
        return buckets

    def _jaccard(self, G: nx.Graph, u, v) -> float:
        
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        union = len(Nu | Nv)
        if union == 0:
            # Both isolated: structurally identical
            return 1.0
        return len(Nu & Nv) / union

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
        """Run the full summarization pipeline. Returns G'."""
        
        if G.number_of_nodes() == 0:
            self.original_graph   = G
            self.summarized_graph = nx.Graph()
            return self.summarized_graph

        self.original_graph = G
        degrees = self._compute_degrees(G)
        buckets = self._build_degree_buckets(degrees, bucket_size)
        find, union = self._make_uf(list(G.nodes()))

        for nodes in buckets.values():
            if len(nodes) < 2:
                continue
            for u, v in combinations(nodes, 2):
                if abs(degrees[u] - degrees[v]) > self.epsilon:
                    continue
                if self._jaccard(G, u, v) >= self.theta:
                    union(u, v)

        groups: dict = defaultdict(list)
        for node in G.nodes():
            groups[find(node)].append(node)

        self.node_to_supernode = {}
        self.supernode_members = {}
        for idx, (root, members) in enumerate(groups.items()):
            label = f"S{idx}"
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
            "method":               "Degree+Jaccard",
            "original_nodes":       V,
            "original_edges":       E,
            "summary_nodes":        Vp,
            "summary_edges":        Ep,
            "compression_ratio":    CR,
            "edge_reduction_ratio": ERR,
            "node_reduction_%":     round((1 - Vp / V) * 100, 2) if V > 0 else 0.0,
            "edge_reduction_%":     round((1 - Ep / E) * 100, 2) if E > 0 else 0.0,
            "modularity_Q":         Q,
            "original_connected":   _safe_is_connected(G),
            "summary_connected":    _safe_is_connected(Gp),
            "theta":                self.theta,
            "epsilon":              self.epsilon,
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

