from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/final_nc"
DISPLAY = {
    "no_context": "w/o Context", "shared_relation": "w/o Relation Specificity",
    "static_execution": "w/o State Conditioning", "operator_off": "w/o Semantic Operator",
}
EVIDENCE = {
    "no_context": ("intervention_context_shuffle.csv", "context_shuffle",
                   "recipient context", "context-health L2 change"),
    "shared_relation": ("intervention_relation_shuffle.csv", "relation_shuffle",
                        "relation-to-edge alignment", "within-target relation-memory shuffle"),
    "static_execution": (None, None, "recipient-state conditioning",
                         "no direct same-checkpoint conditioner-off intervention was required; inspect measured bilinear correction and execution variation"),
    "operator_off": ("intervention_operator_off.csv", "operator_off",
                     "relation-conditioned source operation", "same-checkpoint Delta=0 intervention"),
}


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, text):
    path.write_text(text, encoding="utf-8")


def mean_std(values):
    values = [float(v) for v in values]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def p(v):
    return f"{v * 100:.2f}%"


def main():
    paired = read(RESULTS / "paired_ablation_deltas.csv")
    paired = [r for r in paired if r["row_type"] == "summary" and r["dataset"] == "ALL_15_PAIRS"]
    paired_lookup = {(r["comparison"].split(" - ")[0], r["metric"]): r for r in paired}
    lines = [
        "# Final NC Mechanism Summary",
        "",
        "All evidence below is descriptive. Trained-ablation contrasts compare separate validation-selected fits on matched seeds. Same-checkpoint interventions perturb a Full checkpoint and are not causal contribution estimates. Three seeds do not support a significance claim.",
        "",
        "| Mechanism | Trained ablation: paired test delta vs Full (mean ± SD) | Win / tie / loss (15 pairs) | Same-checkpoint evidence |",
        "|---|---:|---:|---|",
    ]
    for variant in ("no_context", "shared_relation", "static_execution", "operator_off"):
        acc = paired_lookup[(variant, "test_acc")]
        f1 = paired_lookup[(variant, "test_macro_f1")]
        ablation = (
            f"Acc {p(float(acc['mean']))} ± {p(float(acc['sd']))}; "
            f"Macro-F1 {p(float(f1['mean']))} ± {p(float(f1['sd']))}"
        )
        wins = f"Acc {acc['wins']}/{acc['ties']}/{acc['losses']}; F1 {f1['wins']}/{f1['ties']}/{f1['losses']}"
        file_name, intervention, _, description = EVIDENCE[variant]
        if file_name:
            rows = [r for r in read(RESULTS / file_name) if r["intervention"] == intervention]
            by_checkpoint = {}
            for row in rows:
                by_checkpoint[(row["dataset"], row["seed"])] = row
            selected = list(by_checkpoint.values())
            logit_mean, logit_sd = mean_std([r["all_node_logit_l2_mean"] for r in selected])
            flip_mean, _ = mean_std([r["test_prediction_flip_rate"] for r in selected])
            test_delta, test_sd = mean_std([r["test_acc_change"] for r in selected])
            evidence = (f"{description}: mean logit L2 {logit_mean:.4g} ± {logit_sd:.4g}; "
                        f"test flip {p(flip_mean)}; test Acc change {p(test_delta)} ± {p(test_sd)}")
        else:
            health = read(RESULTS / "execution_health.csv")
            correction, correction_sd = mean_std([r["bilinear_correction_relative_norm"] for r in health])
            code_var, code_var_sd = mean_std([r["execution_code_variance"] for r in health])
            evidence = (f"{description}: Full mean ||xi-r||/||r|| {correction:.4g} ± {correction_sd:.4g}; "
                        f"execution-code variance {code_var:.4g} ± {code_var_sd:.4g}")
        lines.append(f"| {DISPLAY[variant]} | {ablation} | {wins} | {evidence} |")
    lines.extend((
        "",
        "Context evidence is based on degree-matched paired Text/Visual context shuffling and real-versus-null context relation changes. Relation-specificity evidence uses within-target cyclic relation-memory shuffling. Operator evidence compares Full with the same checkpoint evaluated at `Delta=0`. State conditioning has no separate same-checkpoint intervention in the frozen protocol; its mechanism audit is the measured bilinear correction, execution-code variation, and the trained `static_execution` contrast.",
        "",
        "Per-dataset and per-seed results are in `paired_ablation_deltas.csv`, `context_health.csv`, `execution_health.csv`, `intervention_context_shuffle.csv`, `intervention_relation_shuffle.csv`, and `intervention_operator_off.csv`.",
    ))
    write(ROOT / "docs/final_nc_mechanism_summary.md", "\n".join(lines) + "\n")
    print("WROTE docs/final_nc_mechanism_summary.md")


if __name__ == "__main__":
    main()
