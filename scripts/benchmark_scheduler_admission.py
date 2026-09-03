#!/usr/bin/env python3
"""Compare legacy queue scanning with incremental decode-block accounting."""

import argparse
import time
from collections.abc import Callable
from typing import TypeVar

from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler

T = TypeVar("T")


def make_request(
    request_id: str,
    prompt_length: int,
    max_tokens: int,
):
    request = InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )
    request.append_generated_token_id(1)
    request.status = RequestStatus.RUNNING
    return request


def legacy_running_reservation(scheduler: Scheduler) -> int:
    reserved_blocks = 0
    running_queue_size = scheduler.running_queue.sync_q.qsize()
    for _ in range(running_queue_size):
        request = scheduler.running_queue.sync_q.get()
        remaining_tokens = (
            request.sampling_params.max_tokens - request.get_num_generated_tokens()
        )
        reserved_blocks += (
            remaining_tokens + scheduler.block_size - 1
        ) // scheduler.block_size
        scheduler.running_queue.sync_q.put(request)
    return reserved_blocks


def legacy_can_accept_request(
    scheduler: Scheduler,
    request: InferenceRequest,
) -> bool:
    total_required_blocks = legacy_running_reservation(scheduler)
    total_length = request.get_prompt_length() + request.sampling_params.max_tokens
    total_required_blocks += (
        total_length + scheduler.block_size - 1
    ) // scheduler.block_size
    return total_required_blocks <= scheduler.cache_manager.get_total_usable_blocks()


def scanned_total_usable_blocks(scheduler: Scheduler) -> int:
    manager = scheduler.cache_manager
    freeable_used_blocks = sum(
        1
        for block_id in manager.used_block_ids
        if manager.blocks[block_id].ref_count == 0
    )
    return manager.get_num_free_blocks() + freeable_used_blocks


def measure(function: Callable[[], T], iterations: int) -> tuple[float, T]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    start = time.perf_counter()
    result = function()
    for _ in range(1, iterations):
        result = function()
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--running-requests", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--prompt-length", type=int, default=257)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    scheduler = Scheduler(num_blocks=4096, block_size=args.block_size)
    for index in range(args.running_requests):
        request = make_request(
            f"request-{index}",
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
        )
        allocation = scheduler.cache_manager.allocate_slots(args.prompt_length)
        if allocation is None:
            raise RuntimeError("Benchmark KV cache is too small for the workload")
        request.block_table, request.slot_mapping = allocation
        scheduler.complete_requests([request])

    candidate = make_request(
        "candidate",
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
    )

    legacy_seconds, legacy_result = measure(
        lambda: legacy_can_accept_request(scheduler, candidate), args.iterations
    )
    optimized_seconds, optimized_result = measure(
        lambda: scheduler.can_accept_request(candidate, 0), args.iterations
    )
    scanned_capacity_seconds, scanned_capacity = measure(
        lambda: scanned_total_usable_blocks(scheduler), args.iterations
    )
    tracked_capacity_seconds, tracked_capacity = measure(
        scheduler.cache_manager.get_total_usable_blocks, args.iterations
    )
    legacy_blocks = legacy_running_reservation(scheduler)
    exact_blocks = scheduler.get_cache_stats()["num_reserved_decode_blocks"]

    print(f"running requests: {args.running_requests}")
    print(f"admission checks: {args.iterations}")
    print(f"prompt length: {args.prompt_length}")
    print(f"max tokens: {args.max_tokens}")
    print(f"block size: {args.block_size}")
    print(f"legacy queue scan: {legacy_seconds:.6f} s")
    print(f"incremental accounting: {optimized_seconds:.6f} s")
    print(f"speedup: {legacy_seconds / optimized_seconds:.2f}x")
    print(f"scanned usable capacity: {scanned_capacity_seconds:.6f} s")
    print(f"tracked usable capacity: {tracked_capacity_seconds:.6f} s")
    print(
        "usable capacity speedup: "
        f"{scanned_capacity_seconds / tracked_capacity_seconds:.2f}x"
    )
    if scanned_capacity != tracked_capacity:
        raise RuntimeError(
            "tracked usable capacity differs from the scanned reference: "
            f"{tracked_capacity} != {scanned_capacity}"
        )
    print(f"legacy reserved blocks: {legacy_blocks}")
    print(f"exact reserved blocks: {exact_blocks}")
    print(f"legacy accepted candidate: {legacy_result}")
    print(f"optimized accepted candidate: {optimized_result}")


if __name__ == "__main__":
    main()
