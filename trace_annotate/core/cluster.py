"""
Near-Duplicate Clustering Engine using Disjoint-Set Union (并查集).
Groups media assets by perceptual dHash with Hamming distance threshold (default <= 10).
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from trace_annotate.core.hasher import hamming_distance


class UnionFind:
    """Disjoint-Set Union structure with path compression and rank optimization."""

    def __init__(self, elements: List[str]):
        self.parent: Dict[str, str] = {x: x for x in elements}
        self.rank: Dict[str, int] = {x: 0 for x in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True


def cluster_by_dhash(
    item_hashes: Dict[str, str],
    hamming_threshold: int = 10,
    prefix: str = "cluster_"
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Cluster items by perceptual hash using Union-Find.
    
    Args:
        item_hashes: mapping of item_id -> 16-hex dHash string
        hamming_threshold: max hamming distance to be considered part of the same cluster
        prefix: prefix for cluster naming
        
    Returns:
        item_to_cluster: mapping of item_id -> cluster_id
        cluster_to_items: mapping of cluster_id -> list of item_ids in that cluster
    """
    valid_items = [item_id for item_id, h in item_hashes.items() if h]
    uf = UnionFind(valid_items)

    n = len(valid_items)
    for i in range(n):
        id_i = valid_items[i]
        h_i = item_hashes[id_i]
        for j in range(i + 1, n):
            id_j = valid_items[j]
            h_j = item_hashes[id_j]
            if hamming_distance(h_i, h_j) <= hamming_threshold:
                uf.union(id_i, id_j)

    # Group by canonical root
    root_groups: Dict[str, List[str]] = {}
    for item_id in valid_items:
        r = uf.find(item_id)
        root_groups.setdefault(r, []).append(item_id)

    sorted_roots = sorted(root_groups.keys(), key=lambda r: (-len(root_groups[r]), r))

    item_to_cluster: Dict[str, str] = {}
    cluster_to_items: Dict[str, List[str]] = {}

    for idx, r in enumerate(sorted_roots, 1):
        c_id = f"{prefix}{idx:04d}"
        members = sorted(root_groups[r])
        cluster_to_items[c_id] = members
        for m in members:
            item_to_cluster[m] = c_id

    # Items without valid hash
    for item_id in item_hashes:
        if item_id not in item_to_cluster:
            item_to_cluster[item_id] = "cluster_unassigned"

    return item_to_cluster, cluster_to_items
