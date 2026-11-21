import csv

from eval.aggregate_ratings import aggregate


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rater_id", "sample_id", "mode", "sign_accuracy_1_5",
                        "naturalness_1_5", "sync_quality_1_5", "comments"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_means_and_ignores_examples(tmp_path):
    src = tmp_path / "ratings.csv"
    _write(src, [
        dict(rater_id="R1", sample_id="E01", mode="text_only",
             sign_accuracy_1_5=2, naturalness_1_5=3, sync_quality_1_5=4, comments=""),
        dict(rater_id="R2", sample_id="E01", mode="text_only",
             sign_accuracy_1_5=4, naturalness_1_5=5, sync_quality_1_5=4, comments=""),
        dict(rater_id="R1", sample_id="E01", mode="context_aware",
             sign_accuracy_1_5=5, naturalness_1_5=4, sync_quality_1_5=5, comments="good"),
        dict(rater_id="R1", sample_id="EXAMPLE_E01", mode="context_aware",
             sign_accuracy_1_5=1, naturalness_1_5=1, sync_quality_1_5=1, comments="ignored"),
        dict(rater_id="R1", sample_id="E02", mode="context_aware",
             sign_accuracy_1_5=9, naturalness_1_5=3, sync_quality_1_5=3, comments="invalid"),
    ])
    out = tmp_path / "summary.csv"
    result = aggregate(str(src), str(out))

    assert result["pending"] is False
    assert result["used"] == 3
    assert result["skipped"] == 2
    by = {(r["mode"], r["metric"]): r for r in result["rows"]}
    assert by[("text_only", "sign_accuracy")]["mean"] == 3.0
    assert by[("text_only", "sign_accuracy")]["n"] == 2
    assert by[("context_aware", "sign_accuracy")]["mean"] == 5.0
    assert out.exists()


def test_aggregate_pending_when_only_examples(tmp_path):
    src = tmp_path / "ratings.csv"
    _write(src, [
        dict(rater_id="R1", sample_id="EXAMPLE_E01", mode="text_only",
             sign_accuracy_1_5=3, naturalness_1_5=3, sync_quality_1_5=3, comments=""),
    ])
    result = aggregate(str(src), str(tmp_path / "summary.csv"))
    assert result["pending"] is True
    assert result["rows"] == []
