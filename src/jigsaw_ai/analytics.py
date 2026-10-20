"""Observed application statistics; uploads have no ground-truth accuracy."""
from collections import Counter
from datetime import datetime, timezone


def application_statistics(storage):
    results = storage.history(2_147_483_647, 0)
    corrections = storage.records("correction", limit=2_147_483_647)
    jobs = storage.records("job", limit=2_147_483_647)
    durations = [row["duration_seconds"] for row in results if row.get("duration_seconds") is not None]
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "completed_reconstructions": len(results),
            "by_scorer": dict(Counter(row["scorer"] for row in results)),
            "by_search": dict(Counter(row["search"] for row in results)),
            "saved_corrections": len(corrections), "corrected_reconstructions": len({row["result_id"] for row in corrections}),
            "background_job_statuses": dict(Counter(row["status"] for row in jobs)),
            "timed_reconstructions": len(durations),
            "mean_processing_seconds": sum(durations) / len(durations) if durations else None,
            "position_accuracy": None, "exact_reconstruction_rate": None,
            "note": "Processing time excludes queue wait, upload and result file writes; local observations are not a benchmark. Accuracy requires ground truth, which is not supplied for uploads."}
