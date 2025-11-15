from collections import defaultdict
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per question (group).
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = defaultdict(list)
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(response, ground_truth)
        for k in r.keys():
            rewards[k].append(r[k])

    rewards["reward"] = torch.tensor(rewards["reward"]).reshape(-1, group_size)
    rewards["format_reward"] = torch.tensor(rewards["format_reward"])
    rewards["answer_reward"] = torch.tensor(rewards["answer_reward"])

    rewards["reward_mean"] = rewards["reward"].mean(dim=-1, keepdim=True)
    advantage = rewards["reward"] - rewards["reward_mean"]

    rewards["reward_std"] = rewards["reward"].std(dim=-1, keepdim=True)
    if normalize_by_std:
        advantage /= rewards["reward_std"] + advantage_eps

    return advantage.flatten(), rewards["reward"].flatten(), rewards


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages.unsqueeze(1) * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    probs_ratio = policy_log_probs / old_log_probs
    l1 = probs_ratio * advantages.unsqueeze(1)
    metadata = {"naive_loss": -l1, "probs_ratio": probs_ratio}
    metadata["clipped_probs_ratio"] = torch.clamp(
        probs_ratio, min=1 - cliprange, max=1 + cliprange
    )
    l2 = metadata["clipped_probs_ratio"] * advantages.unsqueeze(1)
    metadata["clipped_loss"] = -l2
    return -min(l1, l2), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    allowed_loss_types = ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    if loss_type not in allowed_loss_types:
        raise ValueError(
            f"loss_type must be one of the values within {allowed_loss_types}",
            allowed_loss_types,
        )
    if loss_type == "no_baseline":
        return (
            compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs),
            {},
        )  # empty metadata
    elif loss_type == "reinforce_with_baseline":
        return (
            compute_naive_policy_gradient_loss(advantages, policy_log_probs),
            {},
        )  # empty metadata
    else:  # grp_clip
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    s = torch.sum(tensor * mask, dim=dim)
    return s / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    policy_graident_loss_for_seq, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    masked_loss_per_accumulation_step = (
        masked_mean(policy_graident_loss_for_seq, response_mask)
        / gradient_accumulation_steps
    )
    return masked_loss_per_accumulation_step, metadata


def grpo_train_loop(
    data_loader: DataLoader,
    model: nn.Module,
    reward_fn: Callable,
    optimizer: torch.optim.Optimizer = torch.optim.AdamW,
    gradient_accumulation_steps: int = 128,
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip"
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float | None = None,
):
    for _ in range(n_grpo_steps):
        for idx, (inputs, labels) in enumerate(data_loader):
            # Forward pass.
            for _ in range(group_size):
                logits = model(inputs)
                log_probs = torch.log_softmax(logits, dim=-1, keepdim=True)
                advantages, raw_rewards, rewards_metadata = (
                    compute_group_normalized_rewards(
                        reward_fn,
                        rollout_responses,
                        repeated_ground_truths,
                        group_size,
                        advantage_eps,
                        use_std_normalization,
                    )
                )
                loss, metadata = grpo_microbatch_train_step(
                    log_probs,
                    response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    raw_rewards,
                    advantages,
                    old_log_probs,
                    cliprange,
                )
            # loss = loss_fn(logits, labels) / gradient_accumulation_steps
            # # Backward pass.
            # loss.backward()
            # if (idx + 1) % gradient_accumulation_steps == 0:
            #     # Update weights every `gradient_accumulation_steps` batches.
            #     optimizer.step()
            #     optimizer.zero_grad()
