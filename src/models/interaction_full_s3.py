from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .interaction_core_v3 import Model as Stage12Model
from .interaction_history_components import (
    HISTORY_TOKEN_LABELS,
    HistoryReadout,
    HistoryReadoutQuery,
    HistoryTokenEncoder,
    IncomingMeanStd,
)


STAGE3_VARIANTS = (
    "terminal", "state_history", "operation_history", "relation_operation_history",
)
STAGE3_TRACE_GROUPS = (
    "stage3_history_token_state",
    "stage3_history_token_transition",
    "stage3_history_token_operation",
    "stage3_query_intrinsic",
    "stage3_query_relation_env",
    "stage3_readout_attention",
    "stage3_history_output",
)


class Model(Stage12Model):
    """M0-S3 ROHC, preserving the v3 Stage-I/II implementation by inheritance."""

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        super().__init__(cfg, data_info)
        mc = cfg.model
        if self.hidden_dim != 256 or self.relation_dim != 64 or self.operator_rank != 32 or self.conditioner_rank != 32 or self.num_interaction_steps != 2:
            raise ValueError("M0-S3 freezes hidden=256, relation=64, ranks=32, and two interaction steps")
        if self.variant != "context_bilinear_absolute":
            raise ValueError("M0-S3 fixes Stage II to context_bilinear_absolute")
        self.stage3_variant = str(mc.stage3_variant).strip().lower()
        if self.stage3_variant not in STAGE3_VARIANTS:
            raise ValueError(f"model.stage3_variant must be one of {STAGE3_VARIANTS}")
        history_cfg = mc.get("history", {})
        if str(history_cfg.get("operation_profile", "mean_std")) != "mean_std":
            raise ValueError("M0-S3 fixes operation_profile=mean_std")
        self.profile_eps = float(history_cfg.get("eps", 1e-8))
        self.history_token_encoder = HistoryTokenEncoder(self.hidden_dim, 2 * self.operator_rank)
        self.history_query = HistoryReadoutQuery(self.hidden_dim, 2 * self.relation_dim)
        self.history_readout = HistoryReadout(self.hidden_dim, int(history_cfg.get("num_heads", 4)))
        self.history_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        if bool(history_cfg.get("zero_init_output", True)):
            nn.init.zeros_(self.history_output.weight)
            nn.init.zeros_(self.history_output.bias)
        self.incoming_mean_std = IncomingMeanStd(self.profile_eps)
        self._stage3_trace_path = str(mc.get("stage3_gradient_trace_path", ""))
        self._stage3_growth_path = str(mc.get("stage3_growth_trace_path", ""))
        self._stage3_epoch = 0
        self._stage3_epoch_open = False
        self._stage3_grad_sums = {name: 0.0 for name in STAGE3_TRACE_GROUPS}
        self._stage3_grad_numels = {name: 0 for name in STAGE3_TRACE_GROUPS}
        self._stage3_parameter_groups = {
            name: self._stage3_group_for_parameter(name) for name, _ in self.named_parameters()
        }
        for name, parameter in self.named_parameters():
            group = self._stage3_parameter_groups[name]
            if group is not None and parameter.requires_grad:
                parameter.register_hook(self._make_stage3_hook(group))

    @staticmethod
    def _stage3_group_for_parameter(name: str) -> str | None:
        if name.startswith(("history_token_encoder.W_H", "history_token_encoder.norm.")) or name.startswith((
            "history_token_encoder.modality_", "history_token_encoder.stage_", "history_token_encoder.NO_OPERATION_",
        )):
            return "stage3_history_token_state"
        if name.startswith("history_token_encoder.W_D"):
            return "stage3_history_token_transition"
        if name.startswith(("history_token_encoder.W_O", "history_token_encoder.NULL_OPERATION_")):
            return "stage3_history_token_operation"
        if name.startswith(("history_query.global_readout_query", "history_query.W_PT", "history_query.W_PV", "history_query.norm")):
            return "stage3_query_intrinsic"
        if name.startswith(("history_query.W_RT", "history_query.W_RV", "history_query.NULL_REL_ENV_")):
            return "stage3_query_relation_env"
        if name.startswith(("history_readout.attention", "history_readout.norm")):
            return "stage3_readout_attention"
        if name.startswith("history_output."):
            return "stage3_history_output"
        return None

    def _make_stage3_hook(self, group: str):
        def collect(grad: torch.Tensor):
            if self._stage3_epoch_open:
                self._stage3_grad_sums[group] += float(grad.detach().float().square().sum().item())
                self._stage3_grad_numels[group] += int(grad.numel())
            return grad
        return collect

    @staticmethod
    def _append_csv(path_string: str, rows: list[dict], fields: list[str]) -> None:
        if not path_string:
            return
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _flush_stage3_trace(self) -> None:
        if not self._stage3_epoch_open:
            return
        gradient_rows = []
        for group in STAGE3_TRACE_GROUPS:
            count = self._stage3_grad_numels[group]
            grad_rms = (self._stage3_grad_sums[group] / count) ** 0.5 if count else 0.0
            squared_norm = 0.0
            for name, parameter in self.named_parameters():
                if self._stage3_parameter_groups.get(name) == group:
                    squared_norm += float(parameter.detach().float().square().sum().item())
            gradient_rows.append({
                "epoch": self._stage3_epoch,
                "group": group,
                "gradient_rms_preclip": grad_rms,
                "gradient_numel": count,
                "parameter_l2_norm": squared_norm ** 0.5,
            })
        weight_norm = float(self.history_output.weight.detach().float().norm())
        self._append_csv(self._stage3_trace_path, gradient_rows,
                         ["epoch", "group", "gradient_rms_preclip", "gradient_numel", "parameter_l2_norm"])
        self._append_csv(self._stage3_growth_path,
                         [{"epoch": self._stage3_epoch, "W_hist_frobenius_norm": weight_norm}],
                         ["epoch", "W_hist_frobenius_norm"])
        self._stage3_epoch_open = False

    def set_epoch(self, epoch: int) -> None:
        self._flush_stage3_trace()
        super().set_epoch(epoch)
        self._stage3_epoch = int(epoch)
        self._stage3_grad_sums = {name: 0.0 for name in STAGE3_TRACE_GROUPS}
        self._stage3_grad_numels = {name: 0 for name in STAGE3_TRACE_GROUPS}
        self._stage3_epoch_open = True

    def _profile(self, edge_values: torch.Tensor, edge_targets: torch.Tensor, num_nodes: int) -> torch.Tensor:
        mean, std = self.incoming_mean_std(edge_values, edge_targets, num_nodes)
        return torch.cat((mean, std), dim=-1)

    def _encode_stage12(self, x: torch.Tensor, edge_index: torch.Tensor | None, *, diagnostics: bool):
        if not self.training:
            self._flush_epoch_trace()
            self._flush_stage3_trace()
        x_text, x_visual = x[:, :self.text_dim], x[:, self.text_dim:]
        h0_text, h0_visual = self.text_projector(x_text), self.visual_projector(x_visual)
        edges, degree = self._graph(edge_index, x.size(0), x.device)
        src, dst = edges
        memory, evidence = self._relation_memory(
            h0_text, h0_visual, src, dst, context_shuffle=False, diagnostics=diagnostics,
        )
        base_text = self._base_relation_code("text", memory)
        base_visual = self._base_relation_code("visual", memory)

        h1_text, h1_visual = None, None
        h2_text, h2_visual = None, None
        profile_text, profile_visual = [], []
        details_text, details_visual = [], []
        state_text, state_visual = h0_text, h0_visual
        for step in range(self.num_interaction_steps):
            # The exact v3 absolute-conditioner execution step is reused. Capturing
            # its modulation exposes differentiable operation profiles to Stage III.
            state_text, detail_text = self._execution_step(
                state_text, h0_text, memory, base_text, src, dst, "text", True, True, step,
            )
            state_visual, detail_visual = self._execution_step(
                state_visual, h0_visual, memory, base_visual, src, dst, "visual", True, True, step,
            )
            profile_text.append(self._profile(detail_text["modulation"], dst, x.size(0)))
            profile_visual.append(self._profile(detail_visual["modulation"], dst, x.size(0)))
            if diagnostics:
                details_text.append(detail_text)
                details_visual.append(detail_visual)
            else:
                del detail_text, detail_visual
            if step == 0:
                h1_text, h1_visual = state_text, state_visual
            else:
                h2_text, h2_visual = state_text, state_visual

        z_term = self.fusion(torch.cat((h2_text, h2_visual), dim=-1))
        relation_env_text = self._profile(base_text, dst, x.size(0))
        relation_env_visual = self._profile(base_visual, dst, x.size(0))
        values = {
            "H0_text": h0_text, "H1_text": h1_text, "H2_text": h2_text,
            "H0_visual": h0_visual, "H1_visual": h1_visual, "H2_visual": h2_visual,
            "final_text": h2_text, "final_visual": h2_visual,
            "terminal_z": z_term, "fused_z": z_term,
            "canonical_edge_index": edges, "degree": degree,
            "relation_memory": memory,
            "base_relation_text": base_text, "base_relation_visual": base_visual,
            "relation_environment_text": relation_env_text,
            "relation_environment_visual": relation_env_visual,
            "operation_profile_text_step0": profile_text[0],
            "operation_profile_text_step1": profile_text[1],
            "operation_profile_visual_step0": profile_visual[0],
            "operation_profile_visual_step1": profile_visual[1],
            "transition_text_step1": h1_text - h0_text,
            "transition_text_step2": h2_text - h1_text,
            "transition_visual_step1": h1_visual - h0_visual,
            "transition_visual_step2": h2_visual - h1_visual,
            **evidence,
        }
        if diagnostics:
            values["operation_modulation"] = {
                f"{m}_step{s}": details[s]["modulation"]
                for m, details in (("text", details_text), ("visual", details_visual))
                for s in range(self.num_interaction_steps)
            }
            for key in (
                "operator_deviation_ratio", "cosine_base_delta", "message_rotation_degrees",
                "message_scale", "message_rotation", "base_message_norm", "delta_message_norm",
                "delta_message", "execution_code", "delta_execution_code",
            ):
                values[key] = {
                    f"{m}_step{s}": details[s][key]
                    for m, details in (("text", details_text), ("visual", details_visual))
                    for s in range(self.num_interaction_steps)
                }
        return values

    def _stage3_readout(
        self,
        stage12: dict,
        *,
        need_weights: bool,
        history_off: bool = False,
        operation_profile_off: bool = False,
        operation_alignment_swap: bool = False,
        relation_env_off: bool = False,
        node_conditioning_off: bool = False,
        global_query: bool = False,
        drop_token: int | None = None,
    ) -> dict:
        n = stage12["H0_text"].size(0)
        actual_profiles = {
            "text": (stage12["operation_profile_text_step0"], stage12["operation_profile_text_step1"]),
            "visual": (stage12["operation_profile_visual_step0"], stage12["operation_profile_visual_step1"]),
        }
        uses_operation = self.stage3_variant in ("operation_history", "relation_operation_history")
        if operation_alignment_swap and not uses_operation:
            raise ValueError("operation-alignment swap is only defined for operation-aware variants")
        token_profiles = {}
        for modality in ("text", "visual"):
            if uses_operation and not operation_profile_off:
                p0, p1 = actual_profiles[modality]
                if operation_alignment_swap:
                    p0, p1 = p1, p0
                token_profiles[modality] = (p0, p1)
            else:
                null_op = self.history_token_encoder.null_operation(modality).to(stage12["H0_text"])
                token_profiles[modality] = (null_op.expand(n, -1), null_op.expand(n, -1))

        history = self.history_token_encoder(
            (stage12["H0_text"], stage12["H1_text"], stage12["H2_text"]),
            (stage12["H0_visual"], stage12["H1_visual"], stage12["H2_visual"]),
            token_profiles["text"], token_profiles["visual"],
        )
        uses_relation_env = self.stage3_variant == "relation_operation_history"
        if uses_relation_env and not relation_env_off:
            env_text = stage12["relation_environment_text"]
            env_visual = stage12["relation_environment_visual"]
        else:
            env_text = self.history_query.NULL_REL_ENV_TEXT.to(stage12["H0_text"]).expand(n, -1)
            env_visual = self.history_query.NULL_REL_ENV_VISUAL.to(stage12["H0_text"]).expand(n, -1)
        query = self.history_query(
            stage12["H0_text"], stage12["H0_visual"], env_text, env_visual,
            node_conditioning=not node_conditioning_off,
            relation_environment=True,
            global_only=global_query,
        )
        history_representation, attention_heads = self.history_readout(
            query, history, need_weights=need_weights, drop_token=drop_token,
        )
        c_hist = self.history_output(history_representation)
        if history_off:
            c_hist = torch.zeros_like(c_hist)
        z = stage12["terminal_z"] + c_hist
        result = {
            "terminal_z": stage12["terminal_z"],
            "history_tokens": history,
            "readout_query": query,
            "history_representation": history_representation,
            "history_branch": c_hist,
            "history_operation_profile_text_step0": token_profiles["text"][0],
            "history_operation_profile_text_step1": token_profiles["text"][1],
            "history_operation_profile_visual_step0": token_profiles["visual"][0],
            "history_operation_profile_visual_step1": token_profiles["visual"][1],
            "fused_z": z,
            "history_token_labels": HISTORY_TOKEN_LABELS,
        }
        if need_weights:
            result["readout_attention_heads"] = attention_heads
            result["readout_attention"] = attention_heads.mean(dim=1)
        return result

    def forward(self, x, edge_index=None, **_kwargs):
        if self.stage3_variant == "terminal":
            if not self.training:
                self._flush_stage3_trace()
            return super().forward(x, edge_index)
        stage12 = self._encode_stage12(x, edge_index, diagnostics=False)
        stage3 = self._stage3_readout(stage12, need_weights=False)
        z = stage3["fused_z"]
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(self, x, edge_index=None, **kwargs):
        if self.stage3_variant == "terminal":
            self._flush_stage3_trace()
            values = super().analyze(x, edge_index)
            values.update({
                "H2_text": values["final_text"], "H2_visual": values["final_visual"],
                "terminal_z": values["fused_z"],
                "history_branch": torch.zeros_like(values["fused_z"]),
                "history_token_labels": HISTORY_TOKEN_LABELS,
                "operation_modulation": values["modulation"],
            })
            return values
        stage12 = self._encode_stage12(x, edge_index, diagnostics=True)
        options = {
            "history_off": bool(kwargs.get("history_off", False)),
            "operation_profile_off": bool(kwargs.get("operation_profile_off", False)),
            "operation_alignment_swap": bool(kwargs.get("operation_alignment_swap", False)),
            "relation_env_off": bool(kwargs.get("relation_env_off", False)),
            "node_conditioning_off": bool(kwargs.get("node_conditioning_off", False)),
            "global_query": bool(kwargs.get("global_query", False)),
            "drop_token": kwargs.get("drop_token"),
        }
        readout = self._stage3_readout(stage12, need_weights=True, **options)
        stage12.update(readout)
        stage12["transition_text_step0"] = torch.zeros_like(stage12["H0_text"])
        stage12["transition_visual_step0"] = torch.zeros_like(stage12["H0_visual"])
        stage12["query_relation_environment_text"] = (
            stage12["relation_environment_text"] if self.stage3_variant == "relation_operation_history"
            and not options["relation_env_off"] else self.history_query.NULL_REL_ENV_TEXT.expand(x.size(0), -1)
        )
        stage12["query_relation_environment_visual"] = (
            stage12["relation_environment_visual"] if self.stage3_variant == "relation_operation_history"
            and not options["relation_env_off"] else self.history_query.NULL_REL_ENV_VISUAL.expand(x.size(0), -1)
        )
        return stage12

    def active_parameter_names(self) -> set[str]:
        base = super().active_parameter_names()
        if self.stage3_variant == "terminal":
            return base
        return base | {name for name, _ in self.named_parameters() if self._stage3_parameter_groups.get(name) is not None}


__all__ = ["Model", "STAGE3_VARIANTS", "STAGE3_TRACE_GROUPS"]
