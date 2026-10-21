from trace_annotate.core.cluster import cluster_by_dhash


def test_cluster_by_dhash():
    # 3 items: item1 and item2 identical, item3 completely different
    hashes = {
        "item1": "0000000000000000",
        "item2": "0000000000000001", # hamming = 1
        "item3": "ffffffffffffffff", # hamming = 64
    }
    item_to_c, c_to_items = cluster_by_dhash(hashes, hamming_threshold=10)

    # item1 and item2 should be in same cluster
    assert item_to_c["item1"] == item_to_c["item2"]
    # item3 should be in a different cluster
    assert item_to_c["item3"] != item_to_c["item1"]
    assert len(c_to_items) == 2
