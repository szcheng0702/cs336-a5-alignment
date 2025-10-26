import json
from typing import Callable, List

from vllm import LLM, SamplingParams

from .drgrpo_grader import r1_zero_reward_fn

QWEN_MATH_1pt5B = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
R1_ZERO_PROMPT = "./prompts/r1_zero.prompt"


def format_prompts(path: str, prompt_path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for d in data:
        d["prompt"] = (
            lines[0]
            + "\n"
            + lines[1].replace("{question}", d["question"])
            + "\n"
            + lines[2]
        )
    return data


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[dict],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Args:
    vllm_model: a name from huggingface model, or a path to the huggingface model
    reward_fn: a callable from [response,groundtruth] to [metrics, scores]
    prompts: list of prompts
    eval_sampling_params: the sampling parameters for the vllm_model to use to generate answers
    """
    prompts = [d["prompt"] for d in data]
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        data[i]["metrics"] = reward_fn(generated_text, data[i]["answer"])

    with open("out.jsonl", "w", encoding="utf-8") as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    # Sample prompts.
    data = format_prompts("../data/gsm8k/test.jsonl", R1_ZERO_PROMPT)

    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # Create an LLM.
    llm = LLM(model=QWEN_MATH_1pt5B)

    evaluate_vllm(llm, r1_zero_reward_fn, data, sampling_params)
