#!/usr/bin/env python3
"""Benchmark staged request admission against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


@dataclass
class RequestResult:
    wave: str
    request_index: int
    started_at: float
    first_token_at: float | None
    finished_at: float
    finish_reason: str | None
    output_tokens: int
    content_chunks: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.error is None
            and self.first_token_at is not None
            and self.finish_reason in {"length", "stop"}
            and self.output_tokens > 0
        )

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started_at) * 1000

    @property
    def tpot_ms(self) -> float | None:
        if self.first_token_at is None or self.output_tokens <= 1:
            return None
        return (
            (self.finished_at - self.first_token_at) * 1000 / (self.output_tokens - 1)
        )

    @property
    def e2e_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000

    def as_dict(self, origin: float) -> dict[str, Any]:
        return {
            "wave": self.wave,
            "request_index": self.request_index,
            "started_s": self.started_at - origin,
            "first_token_s": (
                None if self.first_token_at is None else self.first_token_at - origin
            ),
            "finished_s": self.finished_at - origin,
            "finish_reason": self.finish_reason,
            "output_tokens": self.output_tokens,
            "content_chunks": self.content_chunks,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "error": self.error,
        }


class FirstTokenBarrier:
    def __init__(self, target: int, total: int):
        if not 0 <= target <= total:
            raise ValueError("barrier target must be between zero and total requests")
        self.target = target
        self.total = total
        self.count = 0
        self.completed_without_first_token = 0
        self.event = asyncio.Event()
        if target == 0:
            self.event.set()

    @property
    def reached(self) -> bool:
        return self.count >= self.target

    @property
    def impossible(self) -> bool:
        return self.total - self.completed_without_first_token < self.target

    def mark_first_token(self) -> None:
        self.count += 1
        if self.reached:
            self.event.set()

    def mark_finished_without_first_token(self) -> None:
        self.completed_without_first_token += 1
        if self.impossible:
            self.event.set()


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def summarize_results(results: Sequence[RequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.succeeded]
    if results:
        started_at = min(result.started_at for result in results)
        finished_at = max(result.finished_at for result in results)
        duration_s = finished_at - started_at
    else:
        duration_s = 0.0

    output_tokens = sum(result.output_tokens for result in successful)
    return {
        "attempted_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "duration_s": duration_s,
        "output_tokens": output_tokens,
        "requests_per_second": (
            len(successful) / duration_s if duration_s > 0 else 0.0
        ),
        "output_tokens_per_second": (
            output_tokens / duration_s if duration_s > 0 else 0.0
        ),
        "ttft_ms": metric_summary(
            [result.ttft_ms for result in successful if result.ttft_ms is not None]
        ),
        "tpot_ms": metric_summary(
            [result.tpot_ms for result in successful if result.tpot_ms is not None]
        ),
        "e2e_ms": metric_summary([result.e2e_ms for result in successful]),
    }


def predict_kv_pressure(
    prompt_tokens: int,
    output_tokens: int,
    block_size: int,
    num_blocks: int,
    wave1_requests: int,
    wave2_requests: int,
) -> dict[str, int]:
    prompt_blocks = math.ceil(prompt_tokens / block_size)
    target_blocks = math.ceil((prompt_tokens + output_tokens) / block_size)
    exact_running_reservation = max(target_blocks - prompt_blocks, 0)

    # The first token has already been sampled when a prefill request enters
    # the running queue, matching the legacy scheduler calculation.
    remaining_output_tokens = max(output_tokens - 1, 0)
    legacy_running_reservation = math.ceil(remaining_output_tokens / block_size)

    blocks_after_wave1_prompts = max(
        num_blocks - wave1_requests * prompt_blocks,
        0,
    )
    candidate_blocks = target_blocks
    legacy_available = max(
        blocks_after_wave1_prompts - wave1_requests * legacy_running_reservation,
        0,
    )
    exact_available = max(
        blocks_after_wave1_prompts - wave1_requests * exact_running_reservation,
        0,
    )

    legacy_capacity = legacy_available // candidate_blocks
    exact_capacity = exact_available // candidate_blocks
    return {
        "prompt_blocks_per_request": prompt_blocks,
        "target_blocks_per_request": target_blocks,
        "legacy_running_reservation_per_request": legacy_running_reservation,
        "exact_running_reservation_per_request": exact_running_reservation,
        "false_reserved_blocks_after_wave1": wave1_requests
        * max(legacy_running_reservation - exact_running_reservation, 0),
        "predicted_wave2_admissions_legacy": min(wave2_requests, legacy_capacity),
        "predicted_wave2_admissions_exact": min(wave2_requests, exact_capacity),
    }


def rendered_prompt_length(tokenizer, content: str) -> int:
    messages = [{"role": "user", "content": content}]
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    return len(input_ids)


def build_exact_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target prompt length must be positive")

    filler_units = ("x ", "hello ", "0 ", "a ")
    max_repetitions = max(target_tokens * 3, 64)
    shortest_length = None

    for unit in filler_units:
        overshoot_count = 0
        for repetitions in range(1, max_repetitions + 1):
            content = (unit * repetitions).rstrip()
            length = rendered_prompt_length(tokenizer, content)
            shortest_length = (
                length if shortest_length is None else min(shortest_length, length)
            )
            if length == target_tokens:
                return content, length
            if length > target_tokens:
                overshoot_count += 1
                if overshoot_count >= 16:
                    break

    raise ValueError(
        "could not construct a chat prompt with exactly "
        f"{target_tokens} tokens; shortest rendered prompt was {shortest_length}"
    )


def count_text_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


async def issue_request(
    client,
    tokenizer,
    model: str,
    prompt_content: str,
    output_tokens: int,
    wave: str,
    request_index: int,
    first_token_barrier: FirstTokenBarrier | None = None,
) -> RequestResult:
    started_at = time.perf_counter()
    first_token_at = None
    finished_at = started_at
    finish_reason = None
    content_chunks = 0
    text_parts = []
    error = None

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_content}],
            max_tokens=output_tokens,
            temperature=1.0,
            top_p=1.0,
            stream=True,
            extra_body={"top_k": 1},
        )
        async for chunk in stream:
            now = time.perf_counter()
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            content = choice.delta.content
            if content:
                text_parts.append(content)
                content_chunks += 1
                if first_token_at is None:
                    first_token_at = now
                    if first_token_barrier is not None:
                        first_token_barrier.mark_first_token()
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                finished_at = now
        if finish_reason is None:
            error = "stream ended without a finish reason"
            finished_at = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        finished_at = time.perf_counter()

    if finish_reason == "length":
        measured_output_tokens = output_tokens
    else:
        measured_output_tokens = count_text_tokens(tokenizer, "".join(text_parts))

    if first_token_at is None and error is None:
        error = "stream produced no content"
    if first_token_at is None and first_token_barrier is not None:
        first_token_barrier.mark_finished_without_first_token()

    return RequestResult(
        wave=wave,
        request_index=request_index,
        started_at=started_at,
        first_token_at=first_token_at,
        finished_at=finished_at,
        finish_reason=finish_reason,
        output_tokens=measured_output_tokens,
        content_chunks=content_chunks,
        error=error,
    )


async def run_warmup(client, tokenizer, args, prompt_content: str) -> None:
    if args.warmup_requests == 0:
        return

    warmup_tokens = min(args.output_tokens, args.warmup_output_tokens)
    tasks = [
        asyncio.create_task(
            issue_request(
                client,
                tokenizer,
                args.model,
                prompt_content,
                warmup_tokens,
                "warmup",
                index,
            )
        )
        for index in range(args.warmup_requests)
    ]
    results = await asyncio.gather(*tasks)
    failures = [result for result in results if not result.succeeded]
    if failures:
        raise RuntimeError(
            f"{len(failures)} warmup requests failed: {failures[0].error}"
        )


async def run_staged_benchmark(
    client,
    tokenizer,
    args,
    prompt_content: str,
) -> tuple[list[RequestResult], float, float, int]:
    barrier = FirstTokenBarrier(
        target=args.wave2_start_after,
        total=args.wave1_requests,
    )
    benchmark_started_at = time.perf_counter()

    wave1_tasks = [
        asyncio.create_task(
            issue_request(
                client,
                tokenizer,
                args.model,
                prompt_content,
                args.output_tokens,
                "wave1",
                index,
                barrier,
            )
        )
        for index in range(args.wave1_requests)
    ]

    try:
        await asyncio.wait_for(barrier.event.wait(), timeout=args.barrier_timeout)
    except asyncio.TimeoutError:
        for task in wave1_tasks:
            task.cancel()
        await asyncio.gather(*wave1_tasks, return_exceptions=True)
        raise RuntimeError(
            "timed out waiting for wave 1 first tokens "
            f"({barrier.count}/{barrier.target})"
        ) from None

    if not barrier.reached:
        for task in wave1_tasks:
            task.cancel()
        await asyncio.gather(*wave1_tasks, return_exceptions=True)
        raise RuntimeError(
            "wave 1 cannot reach the first-token barrier "
            f"({barrier.count}/{barrier.target} first tokens; "
            f"{barrier.completed_without_first_token} requests ended before one)"
        )

    wave2_started_at = time.perf_counter()
    wave2_tasks = [
        asyncio.create_task(
            issue_request(
                client,
                tokenizer,
                args.model,
                prompt_content,
                args.output_tokens,
                "wave2",
                index,
            )
        )
        for index in range(args.wave2_requests)
    ]

    results = await asyncio.gather(*wave1_tasks, *wave2_tasks)
    return results, benchmark_started_at, wave2_started_at, barrier.count


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def collect_environment() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "benchmark_git_revision": command_output(
            ["git", "rev-parse", "HEAD"], repository
        ),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_summary(scope: str, summary: dict[str, Any]) -> None:
    print(
        f"{scope:<8} "
        f"ok={summary['successful_requests']}/{summary['attempted_requests']} "
        f"TTFT p50/p95/p99={format_metric(summary['ttft_ms']['p50'])}/"
        f"{format_metric(summary['ttft_ms']['p95'])}/"
        f"{format_metric(summary['ttft_ms']['p99'])} ms "
        f"TPOT p50={format_metric(summary['tpot_ms']['p50'])} ms "
        f"E2E p95={format_metric(summary['e2e_ms']['p95'])} ms "
        f"throughput={summary['output_tokens_per_second']:.2f} tok/s "
        f"RPS={summary['requests_per_second']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-wave serving benchmark that exposes KV admission behavior."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="default")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument(
        "--server-revision",
        default=None,
        help="Git revision of the server under test, recorded in JSON output",
    )
    parser.add_argument("--prompt-tokens", type=int, default=257)
    parser.add_argument("--output-tokens", type=int, default=240)
    parser.add_argument("--wave1-requests", type=int, default=256)
    parser.add_argument("--wave2-requests", type=int, default=128)
    parser.add_argument("--wave2-start-after", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--barrier-timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=768)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = Path(args.model_path.rstrip("/")).name
    if args.wave2_start_after is None:
        args.wave2_start_after = args.wave1_requests

    positive_fields = (
        "prompt_tokens",
        "output_tokens",
        "wave1_requests",
        "block_size",
        "num_blocks",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.wave2_requests < 0 or args.warmup_requests < 0:
        parser.error("request counts cannot be negative")
    if args.warmup_output_tokens <= 0:
        parser.error("--warmup-output-tokens must be positive")
    if args.barrier_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if not 0 <= args.wave2_start_after <= args.wave1_requests:
        parser.error("--wave2-start-after must be between 0 and wave 1 size")
    return args


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import httpx
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark dependencies are missing; install openai, httpx, and transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    prompt_content, measured_prompt_tokens = build_exact_prompt(
        tokenizer,
        args.prompt_tokens,
    )
    pressure = predict_kv_pressure(
        prompt_tokens=measured_prompt_tokens,
        output_tokens=args.output_tokens,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        wave1_requests=args.wave1_requests,
        wave2_requests=args.wave2_requests,
    )

    max_connections = max(
        args.wave1_requests + args.wave2_requests,
        args.warmup_requests,
        10,
    )
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(args.request_timeout),
    )
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=0,
        http_client=http_client,
    )

    try:
        await client.models.list()
        print(
            f"Warmup: {args.warmup_requests} requests; "
            f"measured prompt length: {measured_prompt_tokens} tokens"
        )
        await run_warmup(client, tokenizer, args, prompt_content)
        print(
            f"Load: wave1={args.wave1_requests}, wave2={args.wave2_requests}, "
            f"wave2 barrier={args.wave2_start_after} first tokens"
        )
        (
            results,
            started_at,
            wave2_started_at,
            barrier_count,
        ) = await run_staged_benchmark(client, tokenizer, args, prompt_content)
    finally:
        await client.close()

    wave1 = [result for result in results if result.wave == "wave1"]
    wave2 = [result for result in results if result.wave == "wave2"]
    summaries = {
        "overall": summarize_results(results),
        "wave1": summarize_results(wave1),
        "wave2": summarize_results(wave2),
    }
    payload = {
        "label": args.label,
        "environment": collect_environment(),
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "model_path": args.model_path,
            "server_revision": args.server_revision,
            "prompt_tokens": measured_prompt_tokens,
            "output_tokens": args.output_tokens,
            "wave1_requests": args.wave1_requests,
            "wave2_requests": args.wave2_requests,
            "wave2_start_after": args.wave2_start_after,
            "warmup_requests": args.warmup_requests,
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
        },
        "pressure_prediction": pressure,
        "wave2_started_s": wave2_started_at - started_at,
        "wave1_first_tokens_at_barrier": barrier_count,
        "summary": summaries,
        "requests": [result.as_dict(started_at) for result in results],
    }

    print("\nPredicted KV admission at the wave 2 boundary:")
    print(json.dumps(pressure, indent=2))
    print("\nMeasured serving results:")
    for scope in ("overall", "wave1", "wave2"):
        print_summary(scope, summaries[scope])
    return payload


def main() -> None:
    args = parse_args()
    try:
        payload = asyncio.run(async_main(args))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
