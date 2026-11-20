"""Pure-PyTorch GRPO trainer with Top-Level and All-Depth Recursive Credit Assignment.

Designed for single-GPU Colab H100/A100 training on Qwen2.5-7B-Instruct + LoRA:
- Computes reference log-probabilities via `model.disable_adapter()` (zero extra VRAM).
- Supports controlled RQ2 ablation between `credit_mode="top_level"` and `"all_depth"`.
- Uses unbiased non-negative Schulman KL estimator and token-level loss accumulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from src.executor import ConductorTurnRecord, RolloutTrace


CreditMode = Literal["top_level", "all_depth"]


@dataclass
class GRPOConfig:
    """Hyperparameters for Conductor GRPO optimization."""

    group_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    grpo_epochs: int = 1
    format_reward: float = 0.2
    cost_lambda: float = 0.0
    reference_cost_usd: float = 0.015
    credit_mode: CreditMode = "all_depth"
    depth_gamma: float = 1.0
    max_seq_len: int = 4096
    temperature: float = 0.8
    max_new_tokens: int = 768


@dataclass
class EncodedTurn:
    """Tokenized prompt + completion tensors for a single Conductor decision."""

    input_ids: torch.Tensor      # shape: (1, P + C)
    prompt_len: int              # P >= 1
    completion_len: int          # C >= 1
    depth: int


def compute_reward(
    trace: RolloutTrace,
    is_correct: bool,
    *,
    format_reward: float = 0.2,
    cost_lambda: float = 0.0,
    reference_cost_usd: float = 0.015,
) -> float:
    """Compute scalar episode reward matching arXiv:2512.04388 + cost-aware extension."""
    if not trace.format_valid:
        return 0.0
    base = format_reward + (1.0 - format_reward) * (1.0 if is_correct else 0.0)
    if cost_lambda <= 0.0 or reference_cost_usd <= 0.0:
        return float(base)
    norm_cost = trace.total_cost_usd / reference_cost_usd
    return float(base - cost_lambda * norm_cost)


def compute_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Compute within-group normalized advantages A_i = (r_i - mean) / (std + eps)."""
    if not rewards:
        return []
    t = torch.tensor(rewards, dtype=torch.float32)
    mean = t.mean()
    std = t.std(unbiased=False)
    if float(std) < eps:
        return [0.0] * len(rewards)
    adv = (t - mean) / (std + eps)
    return [float(x) for x in adv.tolist()]


def compute_completion_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log-probabilities strictly over completion tokens [prompt_len:].

    Args:
        model: Causal LM returning object with `.logits` of shape (B, T, V).
        input_ids: Tensor of shape (B, T) with T > prompt_len >= 1.
        prompt_len: Number of prompt tokens to mask out.

    Returns:
        Tensor of shape (B, T - prompt_len) containing log p(x_t | x_<t).
    """
    seq_len = int(input_ids.shape[1])
    if prompt_len < 1 or seq_len <= prompt_len:
        raise ValueError(
            f"Invalid sequence lengths: seq_len={seq_len}, prompt_len={prompt_len}."
        )

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # (B, T, V)

    # Token at index t is predicted by logits at index t - 1
    pred_logits = logits[:, prompt_len - 1 : seq_len - 1, :].float()
    target_ids = input_ids[:, prompt_len:seq_len]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp


def schulman_kl_divergence(
    cur_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Unbiased, non-negative Schulman KL estimator: exp(log r) - log r - 1."""
    log_ratio = (ref_logprobs - cur_logprobs).clamp(min=-20.0, max=20.0)
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_turn_loss(
    *,
    cur_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    advantage: float,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped GRPO surrogate + KL loss for one Conductor turn."""
    log_ratio = (cur_logprobs - old_logprobs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    adv_t = torch.tensor(advantage, dtype=cur_logprobs.dtype, device=cur_logprobs.device)

    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    pg_loss = -torch.min(surr1, surr2).mean()

    if ref_logprobs is not None and kl_beta > 0.0:
        kl = schulman_kl_divergence(cur_logprobs, ref_logprobs).mean()
        total = pg_loss + kl_beta * kl
        kl_val = float(kl.detach().item())
    else:
        total = pg_loss
        kl_val = 0.0

    with torch.no_grad():
        clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean().item())

    stats = {
        "loss": float(total.detach().item()),
        "pg_loss": float(pg_loss.detach().item()),
        "kl": kl_val,
        "clip_frac": clip_frac,
    }
    return total, stats


def encode_conductor_turn(
    tokenizer: Any,
    turn: ConductorTurnRecord,
    *,
    max_seq_len: int = 4096,
    device: torch.device | str = "cpu",
) -> EncodedTurn | None:
    """Tokenize a ConductorTurnRecord cleanly into prompt + completion IDs."""
    prompt_text = tokenizer.apply_chat_template(
        turn.messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    comp_ids = tokenizer.encode(turn.completion_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        comp_ids = comp_ids + [int(tokenizer.eos_token_id)]

    if not prompt_ids or not comp_ids:
        return None

    # Truncate prompt from the left if total sequence exceeds max_seq_len
    total_len = len(prompt_ids) + len(comp_ids)
    if total_len > max_seq_len:
        keep_prompt = max(64, max_seq_len - len(comp_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
        max_comp = max_seq_len - len(prompt_ids)
        comp_ids = comp_ids[:max_comp]

    if not comp_ids:
        return None

    full_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    return EncodedTurn(
        input_ids=full_ids,
        prompt_len=len(prompt_ids),
        completion_len=len(comp_ids),
        depth=turn.depth,
    )


def select_turns_for_credit(
    trace: RolloutTrace,
    credit_mode: CreditMode,
) -> list[ConductorTurnRecord]:
    """Select which Conductor turns receive GRPO gradients (RQ2 core switch)."""
    if not trace.conductor_turns:
        return []
    if credit_mode == "top_level":
        return trace.conductor_turns[:1]
    if credit_mode == "all_depth":
        return list(trace.conductor_turns)
    raise ValueError(f"Unknown credit_mode: {credit_mode}")


class ConductorGRPOTrainer:
    """Single-GPU GRPO trainer for LoRA-adapted Conductor models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: GRPOConfig,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # pi_old is scored in eval mode and pi_theta in train mode: any active dropout would make
        # the importance ratio != 1 at the very first step. Zero it (LoRA dropout included).
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Model has no trainable parameters (did you attach LoRA?).")
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0

    @torch.no_grad()
    def _compute_ref_logprobs(self, encoded: EncodedTurn) -> torch.Tensor | None:
        if self.config.kl_beta <= 0.0:
            return None
        if not hasattr(self.model, "disable_adapter"):
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.disable_adapter():
                return compute_completion_logprobs(
                    self.model, encoded.input_ids, encoded.prompt_len
                ).detach()
        finally:
            if was_training:
                self.model.train()

    def update_from_groups(
        self,
        groups: list[tuple[list[RolloutTrace], list[float]]],
    ) -> dict[str, float]:
        """Run GRPO update over a batch of `(traces, rewards)` groups.

        Args:
            groups: List of length B (prompts per batch), each containing G traces and G rewards.

        Returns:
            Dictionary of step optimization metrics.
        """
        # Prepare turn-level training items: (encoded, advantage, depth_weight, old_lp, ref_lp)
        items: list[tuple[EncodedTurn, float, float, torch.Tensor, torch.Tensor | None]] = []
        skipped_zero_var = 0

        self.model.eval()
        for traces, rewards in groups:
            advantages = compute_group_advantages(rewards)
            if all(abs(a) < 1e-8 for a in advantages):
                skipped_zero_var += 1
                continue

            for trace, adv in zip(traces, advantages):
                if abs(adv) < 1e-8:
                    continue
                turns = select_turns_for_credit(trace, self.config.credit_mode)
                if not turns:
                    continue
                # Normalize per-turn weight so deep recursive trajectories don't overwhelm shallow ones
                turn_scale = 1.0 / len(turns)
                for turn in turns:
                    enc = encode_conductor_turn(
                        self.tokenizer,
                        turn,
                        max_seq_len=self.config.max_seq_len,
                        device=self.device,
                    )
                    if enc is None:
                        continue
                    with torch.no_grad():
                        old_lp = compute_completion_logprobs(
                            self.model, enc.input_ids, enc.prompt_len
                        ).detach()
                    ref_lp = self._compute_ref_logprobs(enc)
                    weight = turn_scale * (self.config.depth_gamma ** enc.depth)
                    items.append((enc, adv, weight, old_lp, ref_lp))

        if not items:
            self.global_step += 1
            return {
                "loss": 0.0,
                "pg_loss": 0.0,
                "kl": 0.0,
                "clip_frac": 0.0,
                "grad_norm": 0.0,
                "num_turns_updated": 0.0,
                "skipped_zero_var_groups": float(skipped_zero_var),
            }

        self.model.train()
        total_weight = sum(w for _, _, w, _, _ in items)
        agg_loss = 0.0
        agg_pg = 0.0
        agg_kl = 0.0
        agg_clip = 0.0
        last_grad_norm = 0.0

        for _ in range(self.config.grpo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for enc, adv, weight, old_lp, ref_lp in items:
                cur_lp = compute_completion_logprobs(
                    self.model, enc.input_ids, enc.prompt_len
                )
                loss, st = grpo_turn_loss(
                    cur_logprobs=cur_lp,
                    old_logprobs=old_lp,
                    ref_logprobs=ref_lp,
                    advantage=adv,
                    clip_eps=self.config.clip_eps,
                    kl_beta=self.config.kl_beta,
                )
                scaled_loss = loss * (weight / max(total_weight, 1e-8))
                scaled_loss.backward()

                agg_loss += st["loss"] * (weight / max(total_weight, 1e-8))
                agg_pg += st["pg_loss"] * (weight / max(total_weight, 1e-8))
                agg_kl += st["kl"] * (weight / max(total_weight, 1e-8))
                agg_clip += st["clip_frac"] * (weight / max(total_weight, 1e-8))

            trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm)
            last_grad_norm = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            self.optimizer.step()

        self.global_step += 1
        denom = float(self.config.grpo_epochs)
        return {
            "loss": agg_loss / denom,
            "pg_loss": agg_pg / denom,
            "kl": agg_kl / denom,
            "clip_frac": agg_clip / denom,
            "grad_norm": last_grad_norm,
            "num_turns_updated": float(len(items)),
            "skipped_zero_var_groups": float(skipped_zero_var),
        }

    def save_checkpoint(self, save_dir: str | Path, *, tag: str = "latest") -> Path:
        """Save LoRA adapter, optimizer state, and trainer metadata (Colab Drive friendly)."""
        ckpt_dir = Path(save_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(ckpt_dir)
        else:
            torch.save(self.model.state_dict(), ckpt_dir / "model.pt")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(ckpt_dir)
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        state = {
            "global_step": self.global_step,
            "config": asdict(self.config),
        }
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        return ckpt_dir

    def load_optimizer_state(self, ckpt_dir: str | Path) -> None:
        """Restore optimizer state and global_step when resuming a Colab run."""
        ckpt_path = Path(ckpt_dir)
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
        state_path = ckpt_path / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.global_step = int(state.get("global_step", 0))
