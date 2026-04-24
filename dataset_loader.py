import networkx as nx
import os

def load_erdos_renyi(n=100, p=0.05, seed=42) -> nx.Graph:
    """Random graph G(n, p) — Erdős–Rényi model."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    print(f"[Dataset] Erdős–Rényi G(n={n}, p={p})  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G

def load_barabasi_albert(n=100, m=2, seed=42) -> nx.Graph:
    """Scale-free graph — Barabási–Albert model."""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    print(f"[Dataset] Barabási–Albert G(n={n}, m={m})  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G

def load_karate_club() -> nx.Graph:
    """Built-in Karate Club graph (Zachary 1977) — 34 nodes, 78 edges."""
    G = nx.karate_club_graph()
    print(f"[Dataset] Karate Club  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G

def load_petersen() -> nx.Graph:
    """Petersen graph — 10 nodes, 15 edges (small demo)."""
    G = nx.petersen_graph()
    print(f"[Dataset] Petersen  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G

def load_from_edgelist(filepath: str) -> nx.Graph:
    """
    Load from a plain edge-list file (one edge per line: u v).
    Compatible with SNAP dataset format after downloading locally.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Edge-list file not found: {filepath}")
    G = nx.read_edgelist(filepath, comments='#', nodetype=int)
    # Keep only largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"[Dataset] Edge-list '{filepath}'  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    return G
