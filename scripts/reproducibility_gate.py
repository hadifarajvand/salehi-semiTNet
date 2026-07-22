#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "reproduction/reference_contract.json"
BASELINE_METRICS = ROOT / "outputs/final/metrics.json"
BASELINE_MANIFEST = ROOT / "outputs/final/run_manifest.json"
PRETRAINED_METRICS = ROOT / "outputs/paper_reproduction/pretrained/metrics.json"
PRETRAINED_MANIFEST = ROOT / "outputs/paper_reproduction/pretrained/run_manifest.json"
TRAINED_METRICS = ROOT / "outputs/paper_reproduction/training/metrics.json"
TRAINED_MANIFEST = ROOT / "outputs/paper_reproduction/training/run_manifest.json"
OUT_JSON = ROOT / "outputs/reproducibility_audit.json"
OUT_MD = ROOT / "outputs/REPRODUCIBILITY_AUDIT.md"

METRICS = ("iou", "dice", "precision", "recall", "f1")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def finite_metric_map(data: dict) -> bool:
    for key in METRICS:
        value = data.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return False
    return True


def metric_deltas(measured: dict, reference: dict) -> dict:
    return {key: float(measured[key]) - float(reference[key]) for key in METRICS}


def validate_baseline(contract: dict) -> dict:
    expected = contract["current_reduced_baseline"]
    metrics = read_json(BASELINE_METRICS)
    manifest = read_json(BASELINE_MANIFEST)
    checks = {
        "metrics_present_and_finite": finite_metric_map(metrics),
        "source_kind_matches": metrics.get("source_kind") == expected["source_kind"],
        "dataset_count_matches": metrics.get("verified_dataset_images") == expected["verified_dataset_images"],
        "split_matches": manifest.get("split") == {
            "train_labeled": expected["used_split"]["labeled_train"],
            "pseudo_unlabeled": expected["used_split"]["pseudo_unlabeled"],
            "test": expected["used_split"]["test"],
        },
    }
    if finite_metric_map(metrics):
        checks["canonical_metrics_match"] = all(
            abs(float(metrics[k]) - float(expected["metrics_percent"][k])) < 1e-9 for k in METRICS
        )
    else:
        checks["canonical_metrics_match"] = False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "allowed_claim": contract["gates"]["G0_reduced_baseline_integrity"]["claim_enabled"],
    }


def validate_pretrained(contract: dict) -> dict:
    if not PRETRAINED_METRICS.is_file() or not PRETRAINED_MANIFEST.is_file():
        return {
            "status": "BLOCKED",
            "reason": "Official-checkpoint evaluation outputs are absent.",
            "required_outputs": [str(PRETRAINED_METRICS.relative_to(ROOT)), str(PRETRAINED_MANIFEST.relative_to(ROOT))],
        }
    measured = read_json(PRETRAINED_METRICS)
    manifest = read_json(PRETRAINED_MANIFEST)
    if not finite_metric_map(measured):
        return {"status": "FAIL", "reason": "Pretrained metrics are missing/non-finite."}

    paper = contract["publication"]["published_metrics_percent"]
    tol = contract["gates"]["G1_official_checkpoint_equivalence"]["absolute_tolerance_percentage_points"]
    deltas = metric_deltas(measured, paper)
    metric_checks = {k: abs(deltas[k]) <= float(tol[k]) for k in METRICS}
    provenance_checks = {
        "checkpoint_sha256": manifest.get("checkpoint_sha256") == contract["author_checkpoint"]["sha256"],
        "test_count": manifest.get("test_count") == contract["publication"]["dataset"]["test"],
        "input_size": manifest.get("input_size") == contract["publication"]["training"]["input_size"],
        "evaluator_declared": bool(manifest.get("evaluator")),
    }
    passed = all(metric_checks.values()) and all(provenance_checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "metric_deltas_percentage_points": deltas,
        "metric_checks": metric_checks,
        "provenance_checks": provenance_checks,
        "allowed_claim": contract["gates"]["G1_official_checkpoint_equivalence"]["claim_enabled"] if passed else None,
    }


def validate_training(contract: dict, g1_status: str) -> dict:
    if not TRAINED_METRICS.is_file() or not TRAINED_MANIFEST.is_file():
        return {
            "status": "BLOCKED",
            "reason": "No full paper-faithful training outputs are present.",
            "dependency": "G1 must pass first." if g1_status != "PASS" else "G1 passed; full training still not executed.",
        }
    metrics = read_json(TRAINED_METRICS)
    manifest = read_json(TRAINED_MANIFEST)
    pub = contract["publication"]
    protocol_checks = {
        "g1_passed": g1_status == "PASS",
        "labeled_train": manifest.get("labeled_train") == pub["dataset"]["labeled_train"],
        "unlabeled_train": manifest.get("unlabeled_train") == pub["dataset"]["unlabeled_train"],
        "test_count": manifest.get("test_count") == pub["dataset"]["test"],
        "input_size": manifest.get("input_size") == pub["training"]["input_size"],
        "iterations": manifest.get("iterations") == pub["training"]["iterations"],
        "optimizer": str(manifest.get("optimizer", "")).lower() == str(pub["training"]["optimizer"]).lower(),
        "base_learning_rate": manifest.get("base_learning_rate") == pub["training"]["base_learning_rate"],
        "publication_compatible_architecture": manifest.get("publication_compatible_architecture") is True,
        "exact_test_identity_manifest": bool(manifest.get("test_identity_manifest_sha256")),
    }
    status = "PASS" if finite_metric_map(metrics) and all(protocol_checks.values()) else "FAIL"
    return {
        "status": status,
        "protocol_checks": protocol_checks,
        "metrics": {k: metrics.get(k) for k in METRICS},
        "allowed_claim": contract["gates"]["G3_training_reproduction"]["claim_enabled"] if status == "PASS" else None,
    }


def render_md(audit: dict) -> str:
    lines = [
        "# SemiTNet Reproducibility Audit",
        "",
        f"Overall classification: **{audit['overall_classification']}**",
        "",
        "This audit deliberately separates artifact integrity from paper equivalence. A PASS on the reduced baseline does not imply reproduction of the published SemiTNet metrics.",
        "",
        "## Evidence gates",
        "",
        f"- G0 reduced baseline integrity: **{audit['G0_reduced_baseline']['status']}**",
        f"- G1 official checkpoint equivalence: **{audit['G1_official_checkpoint']['status']}**",
        f"- G2/G3 paper-faithful training: **{audit['G3_training_reproduction']['status']}**",
        "",
        "## Current defensible statement",
        "",
        audit["defensible_statement"],
        "",
        "## Current reduced-run gap versus paper (percentage points)",
        "",
        "| Metric | Paper | Reduced run | Difference |",
        "|---|---:|---:|---:|",
    ]
    for key in METRICS:
        p = audit["paper_metrics"][key]
        b = audit["reduced_metrics"][key]
        d = b - p
        lines.append(f"| {key.upper()} | {p:.2f} | {b:.2f} | {d:+.2f} |")
    lines += [
        "",
        "## Claim policy",
        "",
        "Until G1 passes, do not claim that this repository reproduces or matches the paper's numerical results. Paper-style figures must remain explicitly separated into published-reference content and measured reduced-simulation content.",
        "",
    ]
    if audit["G1_official_checkpoint"]["status"] == "BLOCKED":
        lines += [
            "## Highest-priority unresolved experiment",
            "",
            "Evaluate the exact author checkpoint (verified SHA256) on the exact 191-image paper test set with a publication-compatible evaluator. This is the fastest decisive test of dataset/evaluator equivalence and must precede expensive retraining.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-paper", action="store_true", help="Exit non-zero unless G1 and G3 both pass")
    args = parser.parse_args()

    contract = read_json(CONTRACT)
    if not contract:
        raise SystemExit("Missing reproduction/reference_contract.json")

    g0 = validate_baseline(contract)
    g1 = validate_pretrained(contract)
    g3 = validate_training(contract, g1["status"])
    paper = contract["publication"]["published_metrics_percent"]
    reduced = contract["current_reduced_baseline"]["metrics_percent"]

    if g0["status"] != "PASS":
        overall = "INVALID_BASELINE"
        statement = "The existing reduced-run evidence is internally inconsistent and must not be delivered until repaired."
    elif g1["status"] != "PASS":
        overall = "REDUCED_BASELINE_VALID_NOT_PAPER_EQUIVALENT"
        statement = contract["claim_policy"]["required_language_for_current_baseline"]
    elif g3["status"] != "PASS":
        overall = "OFFICIAL_CHECKPOINT_REPRODUCED_TRAINING_NOT_YET_REPRODUCED"
        statement = "The official checkpoint evaluation is reproduced within tolerance; independent paper-faithful retraining is not yet established."
    else:
        overall = "PAPER_REPRODUCTION_EVIDENCE_COMPLETE"
        statement = "Official-checkpoint equivalence and a paper-faithful training reproduction both pass the declared evidence gates."

    audit = {
        "overall_classification": overall,
        "G0_reduced_baseline": g0,
        "G1_official_checkpoint": g1,
        "G3_training_reproduction": g3,
        "paper_metrics": paper,
        "reduced_metrics": reduced,
        "defensible_statement": statement,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2) + "\n")
    OUT_MD.write_text(render_md(audit) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"[audit] {OUT_JSON.relative_to(ROOT)}")
    print(f"[audit] {OUT_MD.relative_to(ROOT)}")

    if g0["status"] != "PASS":
        raise SystemExit(2)
    if args.strict_paper and (g1["status"] != "PASS" or g3["status"] != "PASS"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
