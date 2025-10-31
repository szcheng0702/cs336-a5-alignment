from typing import List

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizerBase
):
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_input_ids, batch_mask = [], []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        input_ids = prompt_ids + output_ids

        # 4️⃣ Construct mask: 0 for prompt, 1 for output
        mask = [0] * len(prompt_ids) + [1] * (len(output_ids))

        batch_input_ids.append(torch.tensor(input_ids))
        batch_mask.append(torch.tensor(mask))

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    batch_mask = torch.nn.utils.rnn.pad_sequence(batch_mask batch_first=True, padding_value=0)

    # labels
    labels = batch_input_ids[:, 1:]
    # slice off the final token
    input_ids = batch_input_ids[:, :-1]
    mask = batch_mask[:,:-1]
    return {"input_ids":input_ids,"labels":labels,"response_mask":mask}

def compute_entropy(logits:torch.Tensor)->torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow.
    """
    logz = torch.logsumexp(logits,dim=-1,keepdim=True)
    log_px = logits - logz
    px = torch.exp(log_px)
    # -torch.sum(px*log_px,dim=2) might be numerically unstable
    return logz - torch.sum(px*logits,dim=-1)

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool)->torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_probs = logits - torch.logsumexp(logits,dim=-1,keepdim=True)
    token_log_probs = torch.gather(log_probs,dim=-1, index = labels).squeeze(-1)
    token_entropy = None
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
    return {"log_probs":token_log_probs, "token_entropy":token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.sum(tensor*mask,dim=dim)/normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.
    Args:
    policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
    SFT policy being trained.
    response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
    prompt/padding.
    gradient_accumulation_steps Number of microbatches per optimizer step.
    normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    
    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss:scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        metadata:Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    """
    loss = (-policy_log_probs*response_mask).sum()/normalize_constant
    sequence_length = policy_log_probs.shape[-1]
    metadata = {"microbatch_loss_seq":loss, "microbatch_loss_per_token":loss/sequence_length}
    return loss/gradient_accumulation_steps, metadata

def log_generations(model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,response_mask: torch.Tensor,gradient_accumulation_steps: int,):
    d = get_response_log_probs(model,input_ids,labels,True)
    loss, metadata = sft_microbatch_train_step(d["log_probs"],response_mask)
    d["loss"] = loss
    print(f"Log probs are: {d["log_probs"]}\n")
    print(f"token entropy is {d["token_entropy"]}\n")
    print(f"microbatch_loss_seq:{d["metadata"]["microbatch_loss_seq"]}\n")
    print(f"microbatch_loss_per_token:{d["metadata"]["microbatch_loss_per_token"]}\n")






