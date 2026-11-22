#!/usr/bin/env python3
"""Matched-seed engineering ablation for the GAN pilot coefficient.

This utility is deliberately framed as a smoke/engineering comparison. It
does not compute narrative-quality metrics and must not be presented as a
Chapter 5 reproduction.
"""
import argparse
import json
import platform
import statistics
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_pipeline import run_pipeline


COEFFICIENTS = (0.0, 0.2)
METRICS = (
    "discriminator_loss",
    "generator_loss",
    "adversarial_score",
    "feature_match_score",
    "combined_reward",
    "replay_items",
)


def stats(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--data", default="examples/demo.jsonl")
    parser.add_argument("--output", default="runs/gan-feature-ablation-2026-11-22")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps-per-stage", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    if args.steps_per_stage < 1:
        parser.error("steps-per-stage must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    runs = []

    for seed in args.seeds:
        seed_results = {}
        for coefficient in COEFFICIENTS:
            label = f"seed-{seed}-fm-{coefficient:.1f}"
            stage_results = run_pipeline(
                args.config,
                args.data,
                output / label,
                steps_per_stage=args.steps_per_stage,
                seed=seed,
                learning_rate=args.learning_rate,
                feature_matching_weight=coefficient,
            )
            gan_steps = [item for item in stage_results if item["stage"] == "adversarial"]
            if len(gan_steps) != args.steps_per_stage:
                raise RuntimeError(
                    f"{label}: expected {args.steps_per_stage} GAN steps, got {len(gan_steps)}"
                )
            # Average the per-step metrics within this run. The final buffer
            # size is taken from the final GAN step.
            averaged = {
                key: statistics.mean(float(item[key]) for item in gan_steps)
                for key in METRICS
                if key != "replay_items"
            }
            averaged["replay_items"] = int(gan_steps[-1]["replay_items"])
            seed_results[coefficient] = averaged
            runs.append(
                {
                    "seed": seed,
                    "feature_matching_weight": coefficient,
                    "steps_per_stage": args.steps_per_stage,
                    "gan_metrics": averaged,
                    "run_dir": str(output / label),
                }
            )

        paired.append(
            {
                "seed": seed,
                "deltas_0.2_minus_0.0": {
                    key: float(seed_results[0.2][key]) - float(seed_results[0.0][key])
                    for key in METRICS
                },
            }
        )

    aggregate = {}
    for coefficient in COEFFICIENTS:
        aggregate[f"weight_{coefficient:.1f}"] = {
            metric: stats(
                run["gan_metrics"][metric]
                for run in runs
                if run["feature_matching_weight"] == coefficient
            )
            for metric in METRICS
        }
    delta_summary = {
        metric: stats(pair["deltas_0.2_minus_0.0"][metric] for pair in paired)
        for metric in METRICS
    }
    result = {
        "schema": "ansg.gan_feature_matching_ablation/1.0",
        "interpretation": "engineering smoke comparison; not a narrative-quality evaluation or Chapter 5 reproduction",
        "data": args.data,
        "config": args.config,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "seeds": args.seeds,
        "steps_per_stage": args.steps_per_stage,
        "coefficient_values": list(COEFFICIENTS),
        "runs": runs,
        "aggregate": aggregate,
        "paired_delta_0.2_minus_0.0": delta_summary,
        "paired_deltas_by_seed": paired,
        "limitations": [
            "The default demo corpus has four synthetic examples and is not ANHC.",
            "Only a few updates per run are performed; estimates are highly noisy.",
            "No blinded human ratings or validated narrative-quality metric is computed.",
            "PPO uses the pipeline's explicitly non-CRAFT surrogate reward.",
            "The 0.2 coefficient is an engineering choice, not specified by Chapter 5.",
        ],
    }
    # Store the exact input hash from the first run manifest for provenance.
    first_manifest = json.loads((Path(runs[0]["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    result["data_sha256"] = first_manifest["source_data"]["sha256"]
    destination = output / "ablation_results.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result_path": str(destination), "aggregate": aggregate,
                      "paired_delta_0.2_minus_0.0": delta_summary}, indent=2))


if __name__ == "__main__":
    main()
