from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .interaction_core_v3 import Model as V3Model, TRACE_GROUPS as V3_TRACE_GROUPS
from .prior_retention_components import (
    ContextAdapter,
    FeatureContextGate,
    PriorRetainingStateUpdate,
)


READOUT_VARIANTS = (
    "terminal",
    "prior_delta_gate",
    "prior_update_add",
    "prior_update_gate",
)
TRACE_GROUPS = tuple(V3_TRACE_GROUPS) + (
    "prior_context_adapter_text",
    "prior_context_adapter_visual",
    "prior_gate_text",
    "prior_gate_visual",
)


class Model(V3Model):
    """Prior-retaining readout over the unchanged M0-Core v3 absolute path."""

    def _install_gradient_hooks(self) -> None:
        # V3 calls this during its constructor, before PRCI modules exist.
        if not getattr(self, "_prior_modules_ready", False):
            return
        self._parameter_groups = {
            name: self._group_for_parameter(name) for name, _ in self.named_parameters()
        }
        self._trace_index = {name: i for i, name in enumerate(TRACE_GROUPS)}
        self._grad_sums = [0.0] * len(TRACE_GROUPS)
        self._grad_numels = [0] * len(TRACE_GROUPS)
        for name, parameter in self.named_parameters():
            group = self._parameter_groups[name]
            if group is None or not parameter.requires_grad:
                continue
            group_idx = self._trace_index[group]

            def collect(grad, idx=group_idx):
                self._grad_sums[idx] += float(grad.detach().float().square().sum().item())
                self._grad_numels[idx] += int(grad.numel())
                return grad

            parameter.register_hook(collect)

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        self._prior_modules_ready = False
        super().__init__(cfg, data_info)
        mc = cfg.model
        # The v3 Stage-II variant is fixed, independently of this readout ablation.
        if self.variant != "context_bilinear_absolute":
            raise ValueError("Prior retention fixes Stage II to context_bilinear_absolute")
        self.readout_variant = str(mc.get("readout_variant", "prior_update_gate")).strip().lower()
        if self.readout_variant not in READOUT_VARIANTS:
            raise ValueError(f"model.readout_variant must be one of {READOUT_VARIANTS}")
        self.text_state_update = PriorRetainingStateUpdate(self.hidden_dim, self.dropout_p)
        self.visual_state_update = PriorRetainingStateUpdate(self.hidden_dim, self.dropout_p)
        self.context_adapter_text = ContextAdapter(self.hidden_dim, self.dropout_p)
        self.context_adapter_visual = ContextAdapter(self.hidden_dim, self.dropout_p)
        self.context_gate_text = FeatureContextGate(self.hidden_dim)
        self.context_gate_visual = FeatureContextGate(self.hidden_dim)

        self._trace_paths = {
            "gradient": str(mc.get("prior_gradient_trace_path", "")),
            "growth": str(mc.get("context_branch_growth_path", "")),
        }
        self._trace_epoch = 0
        self._trace_open = False
        self._growth_started = False
        self._prior_modules_ready = True
        self._install_gradient_hooks()

    def _group_for_parameter(self, name: str) -> str | None:
        if name.startswith("context_adapter_text."):
            return "prior_context_adapter_text"
        if name.startswith("context_adapter_visual."):
            return "prior_context_adapter_visual"
        if name.startswith("context_gate_text."):
            return "prior_gate_text"
        if name.startswith("context_gate_visual."):
            return "prior_gate_visual"
        return super()._group_for_parameter(name)

    @staticmethod
    def _append_trace(path_string: str, fieldnames: list[str], rows: list[dict]) -> None:
        if not path_string:
            return
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _growth_values(self) -> list[dict]:
        rows = []
        for modality in ("text", "visual"):
            adapter = getattr(self, f"context_adapter_{modality}")
            gate = getattr(self, f"context_gate_{modality}")
            rows.append({
                "epoch": self._trace_epoch,
                "modality": modality,
                "context_output_weight_frobenius": float(adapter.output.weight.detach().float().norm().item()),
                "context_output_bias_l2": float(adapter.output.bias.detach().float().norm().item()),
                "gate_weight_frobenius": float(gate.linear.weight.detach().float().norm().item()),
                "gate_bias_l2": float(gate.linear.bias.detach().float().norm().item()),
            })
        return rows

    def _flush_epoch_trace(self) -> None:
        if not getattr(self, "_trace_open", False):
            return
        rows = []
        for idx, group in enumerate(TRACE_GROUPS):
            n = self._grad_numels[idx]
            grad_rms = math.sqrt(self._grad_sums[idx] / n) if n else 0.0
            sq_norm = 0.0
            for name, parameter in self.named_parameters():
                if self._parameter_groups.get(name) == group:
                    sq_norm += float(parameter.detach().float().square().sum().item())
            rows.append({
                "epoch": self._trace_epoch,
                "group": group,
                "gradient_rms_preclip": grad_rms,
                "gradient_numel": n,
                "parameter_l2_norm": math.sqrt(sq_norm),
            })
        self._append_trace(
            self._trace_paths["gradient"],
            ["epoch", "group", "gradient_rms_preclip", "gradient_numel", "parameter_l2_norm"],
            rows,
        )
        self._append_trace(
            self._trace_paths["growth"],
            ["epoch", "modality", "context_output_weight_frobenius", "context_output_bias_l2",
             "gate_weight_frobenius", "gate_bias_l2"],
            self._growth_values(),
        )
        self._trace_open = False

    def set_epoch(self, epoch: int) -> None:
        if self._trace_open:
            self._flush_epoch_trace()
        elif not self._growth_started:
            self._trace_epoch = 0
            self._append_trace(
                self._trace_paths["growth"],
                ["epoch", "modality", "context_output_weight_frobenius", "context_output_bias_l2",
                 "gate_weight_frobenius", "gate_bias_l2"],
                self._growth_values(),
            )
            self._growth_started = True
        self._trace_epoch = int(epoch)
        self._grad_sums = [0.0] * len(TRACE_GROUPS)
        self._grad_numels = [0] * len(TRACE_GROUPS)
        self._trace_open = True

    def _encode_prior(
        self,
        x,
        edge_index,
        *,
        operator_enabled=True,
        diagnostics=False,
        relation_shuffle=False,
        context_shuffle=False,
        conditioner_enabled=True,
        step1_dynamic_off=False,
        frozen_query=False,
        context_off=False,
        force_gate_one=False,
        prior_bypass_off=False,
        context_source_override=None,
    ):
        if not self.training:
            self._flush_epoch_trace()
        x_t, x_v = x[:, :self.text_dim], x[:, self.text_dim:]
        h0_t, h0_v = self.text_projector(x_t), self.visual_projector(x_v)
        edges, degree = self._graph(edge_index, x.size(0), x.device)
        src, dst = edges
        memory, evidence = self._relation_memory(h0_t, h0_v, src, dst, context_shuffle, diagnostics)
        if relation_shuffle and src.numel():
            positions_by_target: dict[int, list[int]] = {}
            for edge_pos, target in enumerate(dst.detach().cpu().tolist()):
                positions_by_target.setdefault(int(target), []).append(edge_pos)
            permutation = list(range(src.numel()))
            for positions in positions_by_target.values():
                if len(positions) > 1:
                    for idx, pos in enumerate(positions):
                        permutation[pos] = positions[(idx + 1) % len(positions)]
            memory = memory[torch.tensor(permutation, dtype=torch.long, device=memory.device)]
        base_t = self._base_relation_code("text", memory)
        base_v = self._base_relation_code("visual", memory)

        capture_updates = diagnostics or self.readout_variant != "terminal"
        self.text_state_update.capture_residual = capture_updates
        self.visual_state_update.capture_residual = capture_updates
        h_t, h_v = h0_t, h0_v
        h1_t = h1_v = None
        update_t, update_v = [], []
        details_t, details_v = [], []
        for step in range(self.num_interaction_steps):
            h_t, detail_t = self._execution_step(
                h_t, h0_t, memory, base_t, src, dst, "text", operator_enabled, diagnostics,
                step, conditioner_enabled, step1_dynamic_off, frozen_query and step == 1,
            )
            if capture_updates:
                update_t.append(self.text_state_update.take_residual())
            h_v, detail_v = self._execution_step(
                h_v, h0_v, memory, base_v, src, dst, "visual", operator_enabled, diagnostics,
                step, conditioner_enabled, step1_dynamic_off, frozen_query and step == 1,
            )
            if capture_updates:
                update_v.append(self.visual_state_update.take_residual())
            if step == 0:
                h1_t, h1_v = h_t, h_v
            if diagnostics:
                details_t.append(detail_t)
                details_v.append(detail_v)

        if self.readout_variant == "terminal":
            final_t, final_v = h_t, h_v
            context_t = context_v = enrich_t = enrich_v = None
            gate_t = gate_v = None
            injected_t = injected_v = None
        else:
            if context_source_override not in (None, "update", "delta"):
                raise ValueError("context_source_override must be None, 'update', or 'delta'")
            if context_source_override is None:
                source = "delta" if self.readout_variant == "prior_delta_gate" else "update"
            else:
                source = context_source_override
            if source == "delta":
                context_t, context_v = h_t - h0_t, h_v - h0_v
            else:
                context_t = sum(update_t) / float(self.num_interaction_steps)
                context_v = sum(update_v) / float(self.num_interaction_steps)
            enrich_t = self.context_adapter_text(context_t)
            enrich_v = self.context_adapter_visual(context_v)
            if self.readout_variant == "prior_update_add":
                gate_t = torch.ones_like(enrich_t)
                gate_v = torch.ones_like(enrich_v)
            else:
                gate_t = self.context_gate_text(h0_t, enrich_t)
                gate_v = self.context_gate_visual(h0_v, enrich_v)
            if force_gate_one:
                gate_t, gate_v = torch.ones_like(gate_t), torch.ones_like(gate_v)
            injected_t, injected_v = gate_t * enrich_t, gate_v * enrich_v
            if context_off:
                injected_t, injected_v = torch.zeros_like(injected_t), torch.zeros_like(injected_v)
            if prior_bypass_off:
                final_t, final_v = injected_t, injected_v
            else:
                final_t, final_v = h0_t + injected_t, h0_v + injected_v

        fused = self.fusion(torch.cat((final_t, final_v), dim=-1))
        values: dict[str, Any] = {
            "H0_text": h0_t, "H0_visual": h0_v,
            "H1_text": h1_t, "H1_visual": h1_v,
            "H2_text": h_t, "H2_visual": h_v,
            "final_text": final_t, "final_visual": final_v, "fused_z": fused,
            "canonical_edge_index": edges, "degree": degree, "relation_memory": memory,
            "base_relation_text": base_t, "base_relation_visual": base_v,
            "C_update_text": sum(update_t) / float(self.num_interaction_steps) if update_t else None,
            "C_update_visual": sum(update_v) / float(self.num_interaction_steps) if update_v else None,
            "C_delta_text": h_t - h0_t, "C_delta_visual": h_v - h0_v,
            "E_text": enrich_t, "E_visual": enrich_v,
            "gate_text": gate_t, "gate_visual": gate_v,
            "R_text": injected_t, "R_visual": injected_v,
            **evidence,
        }
        if diagnostics:
            for step in range(self.num_interaction_steps):
                values[f"U_text_step{step}"] = update_t[step]
                values[f"U_visual_step{step}"] = update_v[step]
                for modality, details in (("text", details_t), ("visual", details_v)):
                    for key, value in details[step].items():
                        values[f"{key}_{modality}_step{step}"] = value
            for key in ("execution_code", "delta_execution_code", "modulation", "conditioner_latent",
                        "conditioner_u_r", "conditioner_u_h", "execution_attention",
                        "delta_message", "operator_deviation_ratio", "base_message_norm",
                        "delta_message_norm", "cosine_base_delta", "message_rotation_degrees",
                        "message_scale", "message_rotation"):
                values[key] = {
                    f"{m}_step{s}": values[f"{key}_{m}_step{s}"]
                    for m in ("text", "visual") for s in range(self.num_interaction_steps)
                }
        self.text_state_update.capture_residual = False
        self.visual_state_update.capture_residual = False
        self.text_state_update._last_residual = None
        self.visual_state_update._last_residual = None
        return fused, values

    def forward(
        self,
        x,
        edge_index=None,
        *,
        operator_enabled=True,
        relation_shuffle=False,
        context_shuffle=False,
        conditioner_enabled=True,
        step1_dynamic_off=False,
        frozen_query=False,
        context_off=False,
        force_gate_one=False,
        prior_bypass_off=False,
        context_source_override=None,
    ):
        z, _ = self._encode_prior(
            x, edge_index, operator_enabled=operator_enabled, diagnostics=False,
            relation_shuffle=relation_shuffle, context_shuffle=context_shuffle,
            conditioner_enabled=conditioner_enabled, step1_dynamic_off=step1_dynamic_off,
            frozen_query=frozen_query, context_off=context_off, force_gate_one=force_gate_one,
            prior_bypass_off=prior_bypass_off, context_source_override=context_source_override,
        )
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(self, x, edge_index=None, **kwargs):
        allowed = {
            "operator_enabled", "relation_shuffle", "context_shuffle", "conditioner_enabled",
            "step1_dynamic_off", "frozen_query", "context_off", "force_gate_one",
            "prior_bypass_off", "context_source_override",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(f"unexpected analyze keyword(s): {unknown}")
        _, values = self._encode_prior(x, edge_index, diagnostics=True, **kwargs)
        return values

    def active_parameter_names(self) -> set[str]:
        names = {name for name, _ in self.named_parameters() if self._group_for_parameter(name) in V3_TRACE_GROUPS}
        if self.readout_variant != "terminal":
            names.update(name for name, _ in self.named_parameters()
                         if name.startswith(("context_adapter_text.", "context_adapter_visual.")))
        if self.readout_variant in ("prior_delta_gate", "prior_update_gate"):
            names.update(name for name, _ in self.named_parameters()
                         if name.startswith(("context_gate_text.", "context_gate_visual.")))
        return names


__all__ = ["Model", "READOUT_VARIANTS", "TRACE_GROUPS"]
