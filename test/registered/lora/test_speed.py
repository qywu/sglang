#!/usr/bin/env python3
"""
Systematic LoRA vs Base model speed test with controlled token lengths
Tests: 1K/4K input tokens × 1K/4K output tokens = 4 scenarios
Includes automatic LoRA adapter loading

SETUP REQUIREMENTS:
==================

1. Download the LoRA model from HuggingFace:
   REPO_NAME=luohuashijieyoufengjun/Qwen3-30B-A3B-Instruct-2507-yimei-lora
   LOCAL_PATH=/scratch/huggingface/Qwen3-30B-A3B-Instruct-2507-yimei-lora
   HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download ${REPO_NAME} --local-dir ${LOCAL_PATH}

2. Start the vLLM server using the Docker setup script:
    https://github.com/togethercomputer/vllm-tomni/pull/2#issuecomment-3716164527

3. The server should be running on localhost:8008 with LoRA support enabled

4. This test will automatically:
   - Load the sql_adapter LoRA from the downloaded model
   - Run systematic performance comparisons across 4 token scenarios
   - Generate detailed performance analysis and comparison tables

USAGE:
======
python3 test_qwen3moe_speed_test.py
"""

import requests
import json
import time
import statistics
from typing import List, Tuple, Dict

API_BASE = "http://localhost:8000/v1"
# Note: load/unload_lora_adapter endpoints are at root, not under /v1
API_ROOT = "http://localhost:8000"

# Model configuration
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LORA_NAME = "sql_adapter"
LORA_PATH = "luohuashijieyoufengjun/Qwen3-30B-A3B-Instruct-2507-yimei-lora"


def load_lora_adapter(lora_name: str, lora_path: str) -> bool:
    """Load a LoRA adapter via API"""
    print(f"🔧 Loading LoRA adapter: {lora_name}")
    print(f"   Path: {lora_path}")

    try:
        response = requests.post(
            f"{API_ROOT}/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )

        if response.status_code == 200:
            print(f"✅ LoRA adapter '{lora_name}' loaded successfully")
            return True
        else:
            print(
                f"❌ Failed to load LoRA adapter: {response.status_code} - {response.text}"
            )
            return False
    except Exception as e:
        print(f"❌ Error loading LoRA adapter: {e}")
        return False


def count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters for most text)"""
    return len(text) // 4


def generate_prompt_with_tokens(target_tokens: int) -> str:
    """Generate a prompt with approximately target_tokens tokens"""

    base_text = """
    Large language models (LLMs) have revolutionized natural language processing and artificial intelligence.
    These models, trained on vast amounts of text data, demonstrate remarkable capabilities in understanding
    and generating human-like text across diverse domains and tasks. The architecture underlying most modern
    LLMs is the Transformer, which was introduced in the groundbreaking paper "Attention Is All You Need" by
    Vaswani et al. The Transformer architecture relies on the self-attention mechanism, which allows the model
    to weigh the importance of different words in a sequence when processing each word, enabling it to capture
    long-range dependencies and contextual relationships more effectively than previous sequential models like
    RNNs and LSTMs. The training process for LLMs involves two main phases: pre-training and fine-tuning.
    During pre-training, the model learns from massive datasets containing billions of tokens, developing
    a broad understanding of language patterns, world knowledge, and reasoning capabilities. This phase
    typically uses unsupervised learning objectives such as next-token prediction or masked language modeling.
    The scale of these models is truly impressive, with parameters ranging from millions to hundreds of billions,
    and the computational resources required for training them are enormous, often requiring specialized
    hardware like GPUs and TPUs distributed across multiple data centers. Fine-tuning is the subsequent phase
    where the pre-trained model is adapted for specific tasks or domains using smaller, task-specific datasets.
    This process allows the model to specialize while retaining the broad knowledge gained during pre-training.
    Various fine-tuning techniques have been developed, including full fine-tuning, where all parameters are
    updated, and parameter-efficient methods like LoRA (Low-Rank Adaptation), which only updates a small
    subset of parameters while keeping the original model frozen. LoRA has gained significant popularity
    because it reduces computational costs and memory requirements while maintaining competitive performance.
    The applications of LLMs are vast and growing, spanning content generation, code synthesis, translation,
    summarization, question answering, conversational AI, and many more. However, these powerful models also
    present challenges, including computational costs, potential biases, hallucinations, and ethical concerns
    about misinformation and misuse. Research continues to address these challenges while pushing the
    boundaries of what these models can achieve."""

    # Repeat and trim to reach target tokens
    current_tokens = count_tokens_approx(base_text)
    repetitions = (target_tokens // current_tokens) + 1
    extended_text = (base_text + " ") * repetitions

    # Trim to target length
    while count_tokens_approx(extended_text) > target_tokens:
        extended_text = extended_text[:-50]  # Remove 50 chars at a time

    # Add instruction
    prompt = f"Based on the following text about large language models, please provide a comprehensive analysis and expand on the key concepts mentioned:\n\n{extended_text}\n\nPlease provide a detailed response:"

    return prompt


def measure_systematic_speed(
    model: str, prompt: str, max_tokens: int, runs: int = 3
) -> Dict:
    """Measure speed with systematic token control"""

    prompt_tokens = count_tokens_approx(prompt)
    print(f"🔬 Testing {model}")
    print(f"   Input: ~{prompt_tokens} tokens, Output: {max_tokens} tokens")

    ttft_times = []
    tps_rates = []
    total_times = []
    actual_output_tokens = []

    for run in range(runs):
        print(f"   Run {run + 1}/{runs}...", end="", flush=True)

        start_time = time.time()

        response = requests.post(
            f"{API_BASE}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Slightly creative but consistent
                "stream": False,
            },
            timeout=120,  # Longer timeout for large outputs
        )

        end_time = time.time()
        total_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            completion_tokens = result["usage"]["completion_tokens"]
            actual_output_tokens.append(completion_tokens)

            # Estimate TTFT as ~5% of total time for large outputs
            estimated_ttft = total_time * 0.05
            generation_time = total_time - estimated_ttft
            tps = completion_tokens / generation_time if generation_time > 0 else 0

            ttft_times.append(estimated_ttft)
            tps_rates.append(tps)
            total_times.append(total_time)

            print(f" ✅ {completion_tokens}tok, {tps:.1f}tps, {total_time:.2f}s")
        else:
            print(f" ❌ Error {response.status_code}")
            continue

        time.sleep(2)  # Pause between runs

    if not ttft_times:
        return {"error": "All runs failed"}

    return {
        "model": model,
        "input_tokens": prompt_tokens,
        "target_output_tokens": max_tokens,
        "actual_output_tokens": statistics.mean(actual_output_tokens),
        "runs": len(ttft_times),
        "ttft_avg": statistics.mean(ttft_times),
        "ttft_std": statistics.stdev(ttft_times) if len(ttft_times) > 1 else 0,
        "tps_avg": statistics.mean(tps_rates),
        "tps_std": statistics.stdev(tps_rates) if len(tps_rates) > 1 else 0,
        "total_time_avg": statistics.mean(total_times),
        "total_time_std": statistics.stdev(total_times) if len(total_times) > 1 else 0,
    }


def run_systematic_comparison():
    """Run systematic 4-scenario comparison"""

    print("🧪 SYSTEMATIC LoRA vs Base Model Speed Test")
    print("=" * 70)
    print("Testing 4 scenarios: 1K/4K input × 1K/4K output tokens")
    print("=" * 70)

    # Load LoRA adapter first
    lora_loaded = load_lora_adapter(
        lora_name=LORA_NAME,
        lora_path=LORA_PATH,
    )

    if not lora_loaded:
        print("❌ Failed to load LoRA adapter. Continuing with base model only...")

    # Test scenarios
    scenarios = [
        {"input_tokens": 1000, "output_tokens": 1000, "name": "1K→1K"},
        {"input_tokens": 1000, "output_tokens": 4000, "name": "1K→4K"},
        {"input_tokens": 4000, "output_tokens": 1000, "name": "4K→1K"},
        {"input_tokens": 4000, "output_tokens": 4000, "name": "4K→4K"},
    ]

    # Model IDs - use colon syntax for LoRA: "base-model:lora-name"
    base_model_id = BASE_MODEL
    lora_model_id = f"{BASE_MODEL}:{LORA_NAME}"

    models = [{"id": base_model_id, "name": "Base Model"}]

    # Only add LoRA model if it was loaded successfully
    if lora_loaded:
        models.append({"id": lora_model_id, "name": "LoRA Model"})

    results = {}

    for i, scenario in enumerate(scenarios):
        scenario_name = scenario["name"]
        print(f"\n📊 Scenario {i + 1}/4: {scenario_name}")
        print(
            f"Input: {scenario['input_tokens']} tokens → Output: {scenario['output_tokens']} tokens"
        )
        print("-" * 50)

        # Generate prompt for this scenario
        prompt = generate_prompt_with_tokens(scenario["input_tokens"])

        results[scenario_name] = {}

        for model in models:
            print(f"\n🔬 {model['name']} ({model['id']})")

            result = measure_systematic_speed(
                model=model["id"],
                prompt=prompt,
                max_tokens=scenario["output_tokens"],
                runs=3,
            )

            results[scenario_name][model["id"]] = result

            if "error" not in result:
                print(
                    f"   📈 TTFT: {result['ttft_avg']:.3f}s (±{result['ttft_std']:.3f})"
                )
                print(f"   ⚡ TPS:  {result['tps_avg']:.1f} (±{result['tps_std']:.1f})")
                print(
                    f"   ⏱️  Total: {result['total_time_avg']:.2f}s (±{result['total_time_std']:.2f})"
                )
                print(
                    f"   📄 Actual output: {result['actual_output_tokens']:.0f} tokens"
                )

    # Generate comprehensive summary only if we have both models
    if lora_loaded and len(models) > 1:
        print("\n" + "=" * 70)
        print("📊 SYSTEMATIC PERFORMANCE SUMMARY")
        print("=" * 70)

        summary_table = []

        for scenario_name, scenario_results in results.items():
            base_result = scenario_results.get(base_model_id, {})
            lora_result = scenario_results.get(lora_model_id, {})

            if "error" not in base_result and "error" not in lora_result:
                ttft_ratio = (
                    lora_result["ttft_avg"] / base_result["ttft_avg"]
                    if base_result["ttft_avg"] > 0
                    else 0
                )
                tps_ratio = (
                    lora_result["tps_avg"] / base_result["tps_avg"]
                    if base_result["tps_avg"] > 0
                    else 0
                )
                total_ratio = (
                    lora_result["total_time_avg"] / base_result["total_time_avg"]
                    if base_result["total_time_avg"] > 0
                    else 0
                )

                summary_table.append(
                    {
                        "scenario": scenario_name,
                        "base_ttft": base_result["ttft_avg"],
                        "lora_ttft": lora_result["ttft_avg"],
                        "ttft_ratio": ttft_ratio,
                        "base_tps": base_result["tps_avg"],
                        "lora_tps": lora_result["tps_avg"],
                        "tps_ratio": tps_ratio,
                        "base_total": base_result["total_time_avg"],
                        "lora_total": lora_result["total_time_avg"],
                        "total_ratio": total_ratio,
                    }
                )

        # Print formatted table
        if summary_table:
            print("\n📋 Performance Comparison Table:")
            print("=" * 90)
            print(
                f"{'Scenario':<8} {'Base TTFT':<10} {'LoRA TTFT':<10} {'Ratio':<6} {'Base TPS':<9} {'LoRA TPS':<9} {'Ratio':<6} {'Speedup':<8}"
            )
            print("-" * 90)

            for row in summary_table:
                speedup = (
                    f"{1 / row['total_ratio']:.2f}x"
                    if row["total_ratio"] > 0
                    else "N/A"
                )
                print(
                    f"{row['scenario']:<8} {row['base_ttft']:.3f}s{'':<3} {row['lora_ttft']:.3f}s{'':<3} {row['ttft_ratio']:.2f}x{'':1} "
                    f"{row['base_tps']:.1f}{'':4} {row['lora_tps']:.1f}{'':4} {row['tps_ratio']:.2f}x{'':1} {speedup:<8}"
                )

            # Analysis
            print("\n🔍 Key Insights:")
            avg_ttft_ratio = statistics.mean(
                [row["ttft_ratio"] for row in summary_table]
            )
            avg_tps_ratio = statistics.mean([row["tps_ratio"] for row in summary_table])
            avg_total_ratio = statistics.mean(
                [row["total_ratio"] for row in summary_table]
            )

            print(
                f"   • LoRA TTFT overhead: {avg_ttft_ratio:.2f}x on average ({(avg_ttft_ratio - 1) * 100:.1f}% slower)"
            )
            print(
                f"   • LoRA TPS efficiency: {avg_tps_ratio:.2f}x on average ({(1 - avg_tps_ratio) * 100:.1f}% slower)"
            )
            print(
                f"   • Overall speed impact: {1 / avg_total_ratio:.2f}x faster with base model"
            )

            # Scaling analysis
            small_scenarios = [row for row in summary_table if "1K" in row["scenario"]]
            large_scenarios = [row for row in summary_table if "4K" in row["scenario"]]

            if small_scenarios and large_scenarios:
                small_tps_ratio = statistics.mean(
                    [row["tps_ratio"] for row in small_scenarios]
                )
                large_tps_ratio = statistics.mean(
                    [row["tps_ratio"] for row in large_scenarios]
                )

                print(
                    f"   • Small token scenarios (1K): {small_tps_ratio:.2f}x TPS ratio"
                )
                print(
                    f"   • Large token scenarios (4K): {large_tps_ratio:.2f}x TPS ratio"
                )

                if abs(small_tps_ratio - large_tps_ratio) > 0.05:
                    if large_tps_ratio > small_tps_ratio:
                        print("   • LoRA scales better with larger contexts")
                    else:
                        print("   • LoRA overhead increases with larger contexts")
                else:
                    print("   • LoRA overhead is consistent across context sizes")
    else:
        print("\n⚠️  LoRA comparison skipped - LoRA adapter not loaded")

    print(f"\n🎯 Test completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == "__main__":
    # Check server availability
    try:
        response = requests.get(f"{API_BASE}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model["id"] for model in models["data"]]
            print(f"✅ Server available. Models: {available_models}")
        else:
            print("❌ Server not responding correctly")
            exit(1)
    except Exception as e:
        print(f"❌ Server error: {e}")
        exit(1)

    # Run the systematic comparison
    results = run_systematic_comparison()