from trace_annotate.core.provenance import ProvenanceTracker, FillSource


def test_provenance_tracker():
    tracker = ProvenanceTracker()
    tracker.set_source("primary_entity", FillSource.HUMAN_CONFIRMED)
    tracker.set_source("brightness", FillSource.AUTO_METRIC)

    assert tracker.is_human_confirmed("primary_entity")
    assert not tracker.is_human_confirmed("brightness")
    assert tracker.get_source("brightness") == "auto_metric"

    d = tracker.to_dict()
    assert d["primary_entity"] == "human_confirmed"
    assert d["brightness"] == "auto_metric"

    summary = ProvenanceTracker.summarize_fills([{"fill": d}])
    assert summary["human_confirmed"] == 1
    assert summary["auto_metric"] == 1
