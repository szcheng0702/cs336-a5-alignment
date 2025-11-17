from typing import Any, Callable, Literal

import torch
import wandb
from datasets import load_dataset
from drgrpo_grader import r1_zero_reward_fn
from einops import rearrange
from forward_pass_helper import get_response_log_probs, tokenize_prompt_and_output
from policy_gradient import compute_group_normalized_rewards, grpo_microbatch_train_step
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm_utils import init_vllm, load_policy_into_vllm_instance

SEED = 42
TRAIN_DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"


def init_wandb(hyperparameters: dict[str, Any]):
    wandb.init(
        project="cs336-alignment",
        name="grpo",
        config={"model": "Qwen2.5-Math-1.5B"} | hyperparameters,
    )


def grpo_train_loop(
    train_dataset: Dataset,
    policy: PreTrainedModel,
    model: LLM,
    reward_fn: Callable,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer | None = None,
    learning_rate=1e-5,
    n_grpo_steps: int = 200,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_params: SamplingParams = SamplingParams(
        temperature=1.0, min_tokens=4, max_tokens=1024
    ),
    epochs_per_rollout_batch: int = 1,  # on-policy
    train_batch_size: int = 256,  # on-policy
    gradient_accumulation_steps: int = 128,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip"
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float | None = None,
):
    # Loss got updated per gradient_accumulation_steps example
    assert (
        train_batch_size % gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    # The rollout loss needs to be divisible by group_size since the data has been duplicated group_size times
    assert (
        rollout_batch_size % group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    # At least one prompt needs to be rolledout for a valid GRPO update
    assert (
        train_batch_size >= group_size
    ), "train_batch_size must be greater than or equal to group_size"

    # Loss backpropagated per microbatch
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=n_prompts_per_rollout_batch, shuffle=True
    )

    if not optimizer:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

    # Setup wandb metrics
    init_wandb(locals())
    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    for step in range(n_grpo_steps):
        load_policy_into_vllm_instance(policy, model)
        for inputs, ground_truths in train_data_loader:
            # Forward pass.
            rollouts = model.generate(
                inputs * group_size, sampling_params=sampling_params
            )  # shape inputs_ids, labels, response_mask: (rollout_batch_size, l)
            rollout_responses = [out.outputs[0].text for out in rollouts]
            repeated_ground_truths = ground_truths * group_size
            # compute rewards/advantages
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

            tokenizer_output = tokenize_prompt_and_output(
                inputs, ground_truths, tokenizer
            )
            # duplicate input_ids, labels, response_mask for group_size times
            batch_input_ids = rearrange(
                tokenizer_output["input_ids"], "b l  -> (r b) l", r=group_size
            )
            batch_labels = rearrange(
                tokenizer_output["labels"], "b l  -> (r b) l", r=group_size
            )
            batch_response_mask = rearrange(
                tokenizer_output["response_mask"], "b l  -> (r b) l", r=group_size
            )
            with torch.no_grad():
                old_log_probs_results = get_response_log_probs(
                    model, batch_input_ids, batch_labels, return_token_entropy=False
                )

            response_masks_chunks = torch.chunk(
                batch_response_mask, chunks=n_microbatches_per_rollout_batch, dim=0
            )
            old_log_pbs_chunks = torch.chunk(
                old_log_pbs_chunks, chunks=n_microbatches_per_rollout_batch, dim=0
            )
            log_probs = get_response_log_probs(
                model, batch_input_ids, batch_labels, return_token_entropy=False
            )
            for epoch in range(epochs_per_rollout_batch):
                loss_for_curr_rollout = 0
                optimizer.zero_grad()
                for mb_idx, (
                    mb_response_mask,
                    mb_old_log_pbs,
                    mb_log_pbs,
                ) in enumerate(
                    zip(
                        response_masks_chunks,
                        old_log_pbs_chunks,
                        log_probs,
                    )
                ):
                    loss, metadata = grpo_microbatch_train_step(
                        mb_log_pbs,
                        mb_response_mask,
                        gradient_accumulation_steps,
                        loss_type,
                        raw_rewards,
                        advantages,
                        mb_old_log_pbs,
                        cliprange,
                    )
                    # Backward pass.
                    loss.backward()
                    loss_for_curr_rollout += loss.item()
                    if (mb_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()

            wandb.log(
                {
                    "train_step": step,
                    "train/rollout_epoch": epoch,
                    "train/loss": loss_for_curr_rollout / group_size,
                }
            )
    wandb.finish()


if __name__ == "__main__":
    device = "cuda:0"
    initialized_policy = AutoModelForCausalLM.from_pretrained(
        "./Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-Math-1.5B")

    initialized_policy.to(device)
    qwen2pt5 = init_vllm(
        model_id="./Qwen2.5-Math-1.5B",
        device=device,
        seed=SEED,
        gpu_memory_utilization=0.85,
    )

    train_dataset = load_dataset(TRAIN_DATASET_NAME)
    grpo_train_loop(
        train_dataset, initialized_policy, qwen2pt5, r1_zero_reward_fn, tokenizer
    )
