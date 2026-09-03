# Paged KV Cache Admission Optimization

## Problem

`Scheduler.can_accept_request()` previously scanned and rotated the complete
running queue for every waiting request. With `R` running requests and `W`
admission candidates in one scheduling step, accounting for running decode
headroom cost `O(R * W)` queue operations.

The previous calculation also rounded every request's remaining decode tokens
up to full blocks without considering unused slots in its current last block.
This could reserve one unnecessary block per running request and defer work that
would fit in the KV cache.

## Change

The scheduler now maintains the aggregate decode-block headroom when requests
enter and leave the running queue. Each request's contribution is calculated
from its target token limit and current block table, so existing unused block
capacity is included.

`get_cache_stats()` exposes `num_reserved_decode_blocks` for diagnostics.

## Correctness Tests

The unit tests cover:

- Precise accounting when the current last block has unused slots.
- Counter updates as a request leaves and re-enters the running queue.
- Counter cleanup for a canceled queued request.
- The public cache statistics field.

Run the tests in an installed InfiniLM development environment:

```bash
python -m unittest discover -s test/llm -v
```

## Control-Plane Microbenchmark

Run:

```bash
python scripts/benchmark_scheduler_admission.py
```

Default workload:

- 256 running requests.
- 10,000 admission checks.
- 257 prompt tokens and up to 128 output tokens per request.
- 256 tokens per KV block.
- Real block allocations in `BlockManager`.

Result on an Apple M4 MacBook Air with 24 GB memory and Python 3.12.14:

| Implementation | Time | Running decode blocks reserved |
| --- | ---: | ---: |
| Legacy queue scan | 1.683887 s | 256 |
| Incremental accounting | 0.067284 s | 0 |

The measured admission-check speedup was `25.03x` for this workload. The exact
calculation reserves no additional decode blocks because all 128 output tokens
fit in the partially used second block of each request.

## Limitations And Next Measurement

This is a scheduler control-plane microbenchmark, not an end-to-end model
throughput result. GPU validation should compare the original and optimized
versions under cache pressure and report request admission rate, TTFT, TPOT,
throughput, and peak KV cache usage. `BlockManager.get_total_usable_blocks()`
still scans used blocks; tracking evictable capacity incrementally is a possible
follow-up if profiling shows it remains significant.
