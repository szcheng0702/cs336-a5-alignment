import torch
import torch.nn.functional as F
from forward_pass_helper import get_response_log_probs, tokenize_prompt_and_output
from transformers import PreTrainedTokenizerBase


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    tokenizer_output = tokenize_prompt_and_output(
        [prompt, prompt], [response_chosen, response_rejected], tokenizer
    )
    log_probs = get_response_log_probs(
        lm,
        tokenizer_output["input_ids"],
        tokenizer_output["labels"],
        return_token_entropy=False,
    )
    ref_log_probs = get_response_log_probs(
        lm_ref,
        tokenizer_output["input_ids"],
        tokenizer_output["labels"],
        return_token_entropy=False,
    )
    # softplus = 1+e^x
    # -log(sigmoid(x)) = log(1+e^(-x)) = softplus(-x)
    return F.softplus(
        -beta
        * (
            log_probs["log_probs"][0, :]
            - ref_log_probs["log_probs"][0, :]
            - log_probs["log_probs"][1, :]
            + ref_log_probs["log_probs"][1, :]
        )
    )
