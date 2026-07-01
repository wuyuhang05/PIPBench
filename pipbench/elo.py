from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preferred_from_result(row: dict, confidence_threshold: float) -> str | None:
    preferred = (
        row.get("result", {})
        .get("comparison_and_choice", {})
        .get("final_decision", {})
        .get("preferred_image")
    )
    confidence = (
        row.get("result", {})
        .get("comparison_and_choice", {})
        .get("final_decision", {})
        .get("confidence")
    )
    if preferred not in {"img1", "img2", "tie"}:
        return None
    if confidence is not None and confidence <= confidence_threshold:
        return "tie"
    if row.get("flipped", False):
        if preferred == "img1":
            return "img2"
        if preferred == "img2":
            return "img1"
    return preferred


def load_matches(compete_data_path: str | Path) -> tuple[dict[int, dict], list[str]]:
    matches = {row["id"]: row for row in read_jsonl(compete_data_path)}
    models = sorted({row["model_A"] for row in matches.values()} | {row["model_B"] for row in matches.values()})
    return matches, models


def majority_vote(rows: Iterable[dict], confidence_threshold: float) -> str | None:
    counter = {"img1": 0, "img2": 0, "tie": 0}
    for row in rows:
        preferred = preferred_from_result(row, confidence_threshold)
        if preferred is not None:
            counter[preferred] += 1
    ordered = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] == 0:
        return None
    if ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def aggregate_pairwise_stats(
    compete_data_path: str | Path,
    result_paths: list[str | Path],
    confidence_threshold: float = 0.7,
    use_majority_vote: bool = False,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    matches, models = load_matches(compete_data_path)
    model_to_idx = {model: idx for idx, model in enumerate(models)}
    n = len(models)
    win_matrix = np.zeros((n, n), dtype=float)
    match_counts = np.zeros((n, n), dtype=int)
    results_by_id: dict[int, list[dict]] = {}
    for path in result_paths:
        for row in read_jsonl(path):
            if row.get("id") in matches:
                results_by_id.setdefault(row["id"], []).append(row)

    for match_id, rows in results_by_id.items():
        match = matches[match_id]
        model_a = match["model_A"]
        model_b = match["model_B"]
        idx_a = model_to_idx[model_a]
        idx_b = model_to_idx[model_b]
        decisions = [majority_vote(rows, confidence_threshold)] if use_majority_vote else [
            preferred_from_result(row, confidence_threshold) for row in rows
        ]
        for preferred in decisions:
            if preferred not in {"img1", "img2", "tie"}:
                continue
            match_counts[idx_a, idx_b] += 1
            match_counts[idx_b, idx_a] += 1
            if preferred == "img1":
                win_matrix[idx_a, idx_b] += 1.0
            elif preferred == "img2":
                win_matrix[idx_b, idx_a] += 1.0
            else:
                win_matrix[idx_a, idx_b] += 0.5
                win_matrix[idx_b, idx_a] += 0.5
    return models, win_matrix, match_counts


def bradley_terry_elo(
    models: list[str],
    win_matrix: np.ndarray,
    match_counts: np.ndarray,
    alpha: float = 400.0,
    base_rating: float = 1500.0,
) -> list[dict]:
    n = len(models)
    x_rows = []
    y_rows = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if match_counts[i, j] == 0:
                continue
            row = np.zeros(n, dtype=float)
            row[i] = 1.0
            row[j] = -1.0
            x_rows.append(row)
            y_rows.append(1)
            weights.append(win_matrix[i, j])
            x_rows.append(row)
            y_rows.append(0)
            weights.append(match_counts[i, j] - win_matrix[i, j])
    if not x_rows:
        raise ValueError("No valid pairwise outcomes found.")
    x = np.vstack(x_rows)
    y = np.asarray(y_rows, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)
    clf = LogisticRegression(fit_intercept=False, C=1e6, solver="lbfgs", max_iter=10000)
    clf.fit(x, y, sample_weight=sample_weight)
    theta = clf.coef_[0]
    logits = x @ theta
    p_hat = 1.0 / (1.0 + np.exp(-logits))
    w_diag = sample_weight * p_hat * (1.0 - p_hat)
    fisher = x.T @ (x * w_diag[:, None])
    cov_theta = np.linalg.pinv(fisher)
    se_theta = np.sqrt(np.clip(np.diag(cov_theta), 0, None))
    elo = theta * (alpha / math.log(10.0))
    se_elo = se_theta * (alpha / math.log(10.0))
    elo = elo - elo.mean() + base_rating
    z = 1.96
    rows = [
        {"model": model, "elo": float(score), "ci95": float(se * z)}
        for model, score, se in zip(models, elo, se_elo)
    ]
    return sorted(rows, key=lambda row: row["elo"], reverse=True)


def win_rate_matrix(models: list[str], win_matrix: np.ndarray, match_counts: np.ndarray) -> list[dict]:
    rows = []
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j or match_counts[i, j] == 0:
                continue
            rows.append(
                {
                    "model": model_i,
                    "opponent": model_j,
                    "wins": float(win_matrix[i, j]),
                    "matches": int(match_counts[i, j]),
                    "win_rate": float(win_matrix[i, j] / match_counts[i, j]),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PIPBench Elo / Bradley-Terry ratings.")
    parser.add_argument("--compete-data", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--majority-vote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models, win_matrix, match_counts = aggregate_pairwise_stats(
        args.compete_data,
        args.results,
        confidence_threshold=args.confidence_threshold,
        use_majority_vote=args.majority_vote,
    )
    elo_rows = bradley_terry_elo(models, win_matrix, match_counts)
    win_rows = win_rate_matrix(models, win_matrix, match_counts)
    (out_dir / "elo_scores.json").write_text(json.dumps(elo_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "win_rates.json").write_text(json.dumps(win_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "elo_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "elo", "ci95"])
        writer.writeheader()
        writer.writerows(elo_rows)
    print(json.dumps(elo_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
