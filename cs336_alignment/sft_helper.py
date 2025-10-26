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
    batch_mask = torch.nn.utils.rnn.pad_sequence(
        batch_mask batch_first=True, padding_value=0)

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