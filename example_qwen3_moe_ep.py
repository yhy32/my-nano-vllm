import argparse
from pathlib import Path

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Generate with Qwen3-MoE expert parallelism.")
    parser.add_argument("model", nargs="?", type=Path, default=Path.home() / "huggingface/Qwen3-30B-A3B")
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    path = str(args.model.expanduser())

    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, tensor_parallel_size=args.gpus, enable_expert_parallel=True,
              enforce_eager=True, max_model_len=512, max_num_batched_tokens=512)

    question = "List the prime numbers below 20."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    output = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=32))[0]
    print(f"Prompt: {question}")
    print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
