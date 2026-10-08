import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Any

# Robust import of HDBSCAN
try:
    import hdbscan
    has_hdbscan = True
except ImportError:
    has_hdbscan = False
    from sklearn.cluster import HDBSCAN as sklearn_hdbscan

def run_hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """
    Groups posts using L2-normalized dense embeddings and HDBSCAN.
    This corresponds to cosine similarity clustering.
    """
    if embeddings.shape[0] < min_cluster_size:
        # Not enough samples to cluster, return all noise (-1)
        return -np.ones(embeddings.shape[0], dtype=int)

    # Normalize vectors to unit length
    normalized = normalize(embeddings, norm="l2")
    
    if has_hdbscan:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean")
            labels = clusterer.fit_predict(normalized)
        except Exception as e:
            print(f"hdbscan failed, falling back to sklearn.cluster.HDBSCAN: {e}")
            from sklearn.cluster import HDBSCAN as sklearn_hdbscan
            clusterer = sklearn_hdbscan(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean")
            labels = clusterer.fit_predict(normalized)
    else:
        from sklearn.cluster import HDBSCAN as sklearn_hdbscan
        clusterer = sklearn_hdbscan(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean")
        labels = clusterer.fit_predict(normalized)
        
    return labels

def detect_bursts(
    df: pd.DataFrame, 
    release_date_str: str, 
    threshold_z: float = 2.5
) -> pd.Series:
    """
    Identifies temporal post volume spikes. 
    Computes a rolling Z-score and flags points exceeding threshold_z.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    # Make sure we have a datetime index for rolling window calculation
    df_temp = df.copy()
    df_temp["created_dt"] = pd.to_datetime(df_temp["created_utc"], unit="s")
    df_temp = df_temp.sort_values("created_dt")
    
    # Bucket post counts in 6-hour intervals
    df_temp.set_index("created_dt", inplace=True)
    counts = df_temp.resample("6h").size()
    
    # Calculate rolling statistics (window = 4 intervals / 24 hours)
    rolling_mean = counts.rolling(window=4, min_periods=1).mean()
    rolling_std = counts.rolling(window=4, min_periods=1).std().fillna(1.0)
    
    # Compute z-score
    z_scores = (counts - rolling_mean) / rolling_std
    burst_intervals = z_scores > threshold_z
    
    # Map back to original dataframe rows
    burst_flags = []
    for idx, row in df.iterrows():
        dt = pd.to_datetime(row["created_utc"], unit="s")
        # Find nearest resampled bucket
        bucket = counts.index[counts.index.get_indexer([dt], method="nearest")[0]]
        burst_flags.append(burst_intervals.loc[bucket])
        
    return pd.Series(burst_flags, index=df.index)

def detect_communities(df: pd.DataFrame) -> Dict[str, int]:
    """
    Builds a networkx reply graph based on author interactions, 
    converts it to igraph, and applies the Leiden algorithm.
    """
    G = nx.Graph()
    
    # Ensure all authors are present
    authors = df["author"].dropna().unique()
    G.add_nodes_from(authors)
    
    # Add reply edges where parent_id matches an existing post's ID
    post_author_map = dict(zip(df["id"], df["author"]))
    
    for _, row in df.iterrows():
        parent_id = row.get("parent_id")
        author = row["author"]
        if pd.notna(parent_id) and pd.notna(author):
            # Check if parent is in map
            parent_author = post_author_map.get(parent_id)
            if parent_author and parent_author != author:
                G.add_edge(author, parent_author)
                
    if G.number_of_edges() == 0:
        # Return individual communities if no edges exist
        return {author: idx for idx, author in enumerate(authors)}
        
    # Convert networkx Graph to igraph Graph
    nodes_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    edges_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    
    ig_g = ig.Graph(len(nodes_list), edges_list)
    
    # Run Leiden community detection
    try:
        partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
        
        community_map = {}
        for comm_idx, members in enumerate(partition):
            for member in members:
                community_map[nodes_list[member]] = comm_idx
        return community_map
    except Exception as e:
        print(f"Warning: leidenalg failed: {e}. Falling back to NetworkX community algorithm.")
        # Fallback to networkx greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        community_map = {}
        for comm_idx, comm in enumerate(communities):
            for author in comm:
                community_map[author] = comm_idx
        return community_map

def run_clustering_stage(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    release_date_str: str
) -> pd.DataFrame:
    """Runs all clustering algorithms and returns the enriched DataFrame."""
    df_out = df.copy()
    
    # 1. Near-duplicate document clusters
    df_out["cluster_id"] = run_hdbscan_clustering(embeddings)
    
    # 2. Burst detection
    df_out["in_burst"] = detect_bursts(df_out, release_date_str)
    
    # 3. Community detection
    comm_map = detect_communities(df_out)
    df_out["community_id"] = df_out["author"].map(comm_map).fillna(-1).astype(int)
    
    return df_out

if __name__ == "__main__":
    # Self-test block
    import time
    # Create fake dataframe
    authors = ["userA", "userB", "userC", "userD", "userE", "userA", "userB"]
    ids = ["id_0", "id_1", "id_2", "id_3", "id_4", "id_5", "id_6"]
    parents = [None, "id_0", "id_1", None, "id_3", "id_1", "id_4"]
    created_utcs = [
        time.time() - 3600*24,
        time.time() - 3600*23,
        time.time() - 3600*22,
        time.time() - 3600*2,
        time.time() - 3600*1.5,
        time.time() - 3600*1,
        time.time()
    ]
    
    test_df = pd.DataFrame({
        "id": ids,
        "author": authors,
        "parent_id": parents,
        "created_utc": created_utcs
    })
    
    # Fake embeddings (min size 5 to cluster, let's make 7 vectors of size 768)
    np.random.seed(42)
    fake_embeddings = np.random.normal(0, 1, (7, 768))
    # Make some vectors very similar
    fake_embeddings[1] = fake_embeddings[0] + 0.01
    fake_embeddings[2] = fake_embeddings[0] - 0.01
    fake_embeddings[5] = fake_embeddings[0] + 0.02
    fake_embeddings[6] = fake_embeddings[0] - 0.02
    
    res_df = run_clustering_stage(test_df, fake_embeddings, "2026-08-18")
    print("Enriched DataFrame:")
    print(res_df[["id", "author", "cluster_id", "in_burst", "community_id"]])
