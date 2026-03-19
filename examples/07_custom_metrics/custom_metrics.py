"""
Netra SDK — Custom Metrics Example

Demonstrates how to use Netra's OpenTelemetry-based metrics pipeline
to emit counters, histograms, and up/down counters with rich attribute
dimensions.  Combines custom metrics with @workflow/@task decorators
to show a realistic AI-service monitoring pattern.

Prerequisites:
    pip install netra-sdk python-dotenv

    export NETRA_API_KEY='your-api-key'

Usage:
    python custom_metrics.py
"""

import asyncio
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

from opentelemetry.metrics import Observation

from netra import Netra
from netra.decorators import agent, task, workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# 1.  Initialise the SDK with metrics enabled
# ---------------------------------------------------------------------------

def init_sdk() -> None:
    Netra.init(
        app_name="custom-metrics-example",
        environment="development",
        trace_content=True,
        enable_metrics=True,
        headers=f"x-api-key={os.getenv('NETRA_API_KEY', 'demo-key')}",
    )
    Netra.set_user_id("metrics_demo_user")
    Netra.set_session_id("metrics_demo_session")
    logger.info("Netra SDK initialised with metrics pipeline enabled")


# ---------------------------------------------------------------------------
# 2.  Create instruments from a named Meter
# ---------------------------------------------------------------------------

def create_instruments():
    """Return a dict of OTel instruments scoped to an 'ai_service' meter."""
    meter = Netra.get_meter("ai_service")

    return {
        # How many LLM requests were made, sliced by model / status
        "llm_requests": meter.create_counter(
            name="ai.llm.requests",
            description="Total LLM inference requests",
            unit="1",
        ),
        # Tokens consumed per request (input + output)
        "token_usage": meter.create_counter(
            name="ai.llm.token_usage",
            description="Total tokens consumed",
            unit="tokens",
        ),
        # End-to-end latency distribution
        "request_latency": meter.create_histogram(
            name="ai.llm.request_latency",
            description="LLM request latency",
            unit="ms",
        ),
        # Cost per request in USD-micros
        "request_cost": meter.create_histogram(
            name="ai.llm.request_cost",
            description="Estimated cost per LLM request",
            unit="USD_micro",
        ),
        # Current number of in-flight requests (gauge-like)
        "inflight_requests": meter.create_up_down_counter(
            name="ai.llm.inflight_requests",
            description="Currently in-flight LLM requests",
            unit="1",
        ),
        # Pending items in the task queue
        "queue_depth": meter.create_up_down_counter(
            name="ai.task_queue.depth",
            description="Items waiting in the task queue",
            unit="1",
        ),
    }


# ---------------------------------------------------------------------------
# 3.  Simulated AI service operations that emit metrics
# ---------------------------------------------------------------------------

MODELS = ["gpt-4o", "claude-sonnet", "gemini-pro"]
COST_PER_1K_TOKENS = {"gpt-4o": 5.0, "claude-sonnet": 3.0, "gemini-pro": 1.25}


@task(name="call_llm")  # type: ignore[arg-type]
def call_llm(
    prompt: str,
    model: str,
    instruments: Dict[str, Any],
) -> Dict[str, Any]:
    """Simulate an LLM call and record metrics around it."""
    attrs = {"model": model}

    instruments["inflight_requests"].add(1, attrs)
    instruments["queue_depth"].add(-1, attrs)

    start = time.perf_counter()
    try:
        latency_ms = random.uniform(80, 600)
        time.sleep(latency_ms / 4000)

        input_tokens = len(prompt.split()) * 2
        output_tokens = random.randint(50, 300)
        total_tokens = input_tokens + output_tokens

        cost_micro = total_tokens / 1000 * COST_PER_1K_TOKENS.get(model, 2.0) * 1000

        if random.random() < 0.05:
            raise RuntimeError("Simulated transient LLM error")

        instruments["llm_requests"].add(1, {**attrs, "status": "success"})
        instruments["token_usage"].add(input_tokens, {**attrs, "direction": "input"})
        instruments["token_usage"].add(output_tokens, {**attrs, "direction": "output"})
        instruments["request_latency"].record(latency_ms, attrs)
        instruments["request_cost"].record(cost_micro, attrs)

        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_micro / 1000, 4),
            "response": f"[simulated {model} response to '{prompt[:40]}…']",
        }

    except Exception as exc:
        instruments["llm_requests"].add(1, {**attrs, "status": "error"})
        elapsed = (time.perf_counter() - start) * 1000
        instruments["request_latency"].record(elapsed, attrs)
        raise exc

    finally:
        instruments["inflight_requests"].add(-1, attrs)


@agent(name="select_model")  # type: ignore[arg-type]
def select_model(prompt: str) -> str:
    """Pick the best model for a given prompt based on length heuristic."""
    if len(prompt) > 200:
        return "gpt-4o"
    if len(prompt) > 80:
        return "claude-sonnet"
    return "gemini-pro"


@workflow(name="ai_inference_workflow")  # type: ignore[arg-type]
def run_inference(
    prompts: List[str],
    instruments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Process a batch of prompts through model selection and LLM calls."""
    logger.info("Starting inference workflow for %d prompts", len(prompts))

    instruments["queue_depth"].add(len(prompts), {"source": "batch"})

    results: List[Dict[str, Any]] = []
    for prompt in prompts:
        model = select_model(prompt)  # type: ignore[misc]
        try:
            result = call_llm(prompt, model, instruments)  # type: ignore[misc]
            results.append(result)
        except Exception as exc:
            logger.warning("Prompt failed: %s", exc)
            results.append({"prompt": prompt[:40], "error": str(exc)})

    successes = sum(1 for r in results if "error" not in r)
    logger.info(
        "Inference workflow complete: %d/%d succeeded", successes, len(results)
    )
    return results


# ---------------------------------------------------------------------------
# 4.  Observable instruments (callback-based gauges)
# ---------------------------------------------------------------------------

_simulated_cache_items = 0


def setup_observable_instruments() -> None:
    """Register observable gauges that are read on each export cycle."""
    meter = Netra.get_meter("ai_service")

    def _cache_size_callback(_):
        global _simulated_cache_items
        _simulated_cache_items = random.randint(100, 500)
        return [Observation(_simulated_cache_items, {"cache": "prompt_cache"})]

    def _gpu_utilization_callback(_):
        return [
            Observation(random.uniform(0.1, 0.95), {"device": "gpu:0"}),
            Observation(random.uniform(0.1, 0.95), {"device": "gpu:1"}),
        ]

    meter.create_observable_gauge(
        name="ai.cache.size",
        description="Current prompt-cache entry count",
        unit="1",
        callbacks=[_cache_size_callback],
    )

    meter.create_observable_gauge(
        name="ai.gpu.utilization",
        description="GPU utilization ratio (0-1)",
        unit="1",
        callbacks=[_gpu_utilization_callback],
    )

    logger.info("Observable gauges registered")


# ---------------------------------------------------------------------------
# 5.  Async example — concurrent requests with shared instruments
# ---------------------------------------------------------------------------

@task(name="async_llm_call")  # type: ignore[arg-type]
async def async_llm_call(
    prompt: str,
    model: str,
    instruments: Dict[str, Any],
) -> Dict[str, Any]:
    """Async variant of call_llm for concurrent fan-out."""
    attrs = {"model": model}
    instruments["inflight_requests"].add(1, attrs)
    instruments["queue_depth"].add(-1, attrs)

    latency_ms = random.uniform(80, 600)
    await asyncio.sleep(latency_ms / 4000)

    input_tokens = len(prompt.split()) * 2
    output_tokens = random.randint(50, 300)
    cost_micro = (input_tokens + output_tokens) / 1000 * COST_PER_1K_TOKENS.get(model, 2.0) * 1000

    instruments["llm_requests"].add(1, {**attrs, "status": "success"})
    instruments["token_usage"].add(input_tokens, {**attrs, "direction": "input"})
    instruments["token_usage"].add(output_tokens, {**attrs, "direction": "output"})
    instruments["request_latency"].record(latency_ms, attrs)
    instruments["request_cost"].record(cost_micro, attrs)
    instruments["inflight_requests"].add(-1, attrs)

    return {
        "model": model,
        "tokens": input_tokens + output_tokens,
        "latency_ms": round(latency_ms, 2),
    }


@workflow(name="async_inference_workflow")  # type: ignore[arg-type]
async def run_async_inference(
    prompts: List[str],
    instruments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fan-out prompts concurrently and collect results."""
    logger.info("Starting async inference for %d prompts", len(prompts))
    instruments["queue_depth"].add(len(prompts), {"source": "async_batch"})

    tasks = [
        async_llm_call(prompt, select_model(prompt), instruments)  # type: ignore[misc]
        for prompt in prompts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = [r for r in results if isinstance(r, dict)]
    logger.info("Async inference complete: %d/%d succeeded", len(ok), len(results))
    return [r if isinstance(r, dict) else {"error": str(r)} for r in results]


# ---------------------------------------------------------------------------
# 6.  Dedicated-meter example — separate meter per domain
# ---------------------------------------------------------------------------

def demonstrate_multiple_meters(instruments: Dict[str, Any]) -> None:
    """Show that different parts of an app can own independent meters."""
    billing_meter = Netra.get_meter("billing_service")
    guardrails_meter = Netra.get_meter("guardrails")

    invoice_total = billing_meter.create_counter(
        name="billing.invoice_total",
        description="Running invoice total in USD-micros",
        unit="USD_micro",
    )
    pii_detections = guardrails_meter.create_counter(
        name="guardrails.pii_detections",
        description="PII detections across inputs",
        unit="1",
    )
    scan_latency = guardrails_meter.create_histogram(
        name="guardrails.scan_latency",
        description="Input-scan latency",
        unit="ms",
    )

    for _ in range(5):
        invoice_total.add(random.randint(500, 5000), {"plan": "pro"})
        pii_detections.add(random.randint(0, 3), {"type": "email"})
        scan_latency.record(random.uniform(1, 15), {"scanner": "presidio"})

    logger.info("Multiple-meter demo complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("  Netra SDK — Custom Metrics Example")
    print("=" * 65)

    # Initialise
    init_sdk()
    instruments = create_instruments()
    setup_observable_instruments()

    # --- Sync batch ---
    print("\n--- Synchronous batch inference ---")
    prompts = [
        "Summarise the key points of this quarterly earnings report.",
        "Translate to French: Hello world",
        "Write a haiku about distributed systems and observability tooling in production.",
        "Explain quantum computing to a five-year-old in simple terms.",
        "Generate a SQL query to find the top 10 customers by revenue this quarter.",
    ]
    sync_results = run_inference(prompts, instruments)  # type: ignore[misc]
    for r in sync_results:
        if "error" in r:
            print(f"  FAIL: {r}")
        else:
            print(
                f"  {r['model']:>15}  "
                f"tokens={r['input_tokens']+r['output_tokens']:<5}  "
                f"latency={r['latency_ms']}ms  "
                f"cost=${r['cost_usd']}"
            )

    # --- Async fan-out ---
    print("\n--- Async concurrent inference ---")
    async_prompts = [f"Async prompt #{i}: tell me something interesting" for i in range(8)]
    async_results = asyncio.run(
        run_async_inference(async_prompts, instruments)  # type: ignore[misc]
    )
    for r in async_results:
        if "error" in r:
            print(f"  FAIL: {r}")
        else:
            print(
                f"  {r['model']:>15}  "
                f"tokens={r['tokens']:<5}  "
                f"latency={r['latency_ms']}ms"
            )

    # --- Multiple meters ---
    print("\n--- Multiple independent meters ---")
    demonstrate_multiple_meters(instruments)
    print("  Billing and guardrails metrics recorded.")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  All metrics have been recorded and will be exported on the")
    print("  next flush cycle (default 60 s) or at shutdown.")
    print("=" * 65)

    Netra.shutdown()
    logger.info("SDK shut down — metrics flushed")


if __name__ == "__main__":
    main()
