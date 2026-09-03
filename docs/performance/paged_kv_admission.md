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

## GPU Serving Benchmark

The repository includes a staged serving benchmark that creates the cache
pressure needed to exercise this change:

```bash
python scripts/benchmark_serving_admission.py --help
```

The default workload uses two request waves:

- Wave 1 starts 256 requests with 257 input tokens and 240 output tokens.
- Wave 2 starts 128 requests after every wave 1 request has produced its first
  token and entered decode.
- With 768 blocks of 256 tokens, the legacy calculation reports 256 additional
  decode blocks for wave 1 and predicts no immediate wave 2 admissions.
- Exact accounting reports no additional wave 1 decode blocks and predicts
  that all 128 wave 2 requests fit.

Use a text model whose tokenizer has a chat template. TinyLlama 1.1B is a
small, documented InfiniLM example that is suitable for a single 24 GB NVIDIA
GPU. Build the NVIDIA InfiniCore backend and install InfiniLM as described in
the repository README before starting the server.

Keep the benchmark script in the optimized checkout and create a detached
worktree for the baseline. This avoids losing the new script when switching to
the old revision:

```bash
export OPTIMIZED_REPO="$(pwd -P)"
export BASELINE_REPO="$(dirname "$OPTIMIZED_REPO")/InfiniLM-baseline"

git worktree add --detach \
  "$BASELINE_REPO" \
  80bb09ecebc9aabf198b9b866a89456bca1df946
git merge-base --is-ancestor \
  81d4e551924ce30fba99245eb8ac819839cb125b HEAD
```

Set `OPTIMIZED_REPO` and `BASELINE_REPO` to these same absolute paths in every
shell used below; shell-local exports are not shared between terminals.

Use a separate virtual environment for each server checkout, especially when
using editable installs. Install InfiniCore and InfiniLM in both environments
using the repository instructions. Install the client-only packages in the
optimized environment:

```bash
"$OPTIMIZED_REPO/.venv/bin/python" -m pip install \
  openai httpx transformers
```

Run the baseline server first. `PYTHONPATH` is explicit so an editable install
from the optimized checkout cannot accidentally supply the baseline's Python
code:

```bash
cd "$BASELINE_REPO"

INFINILM_MAX_NUM_BATCHED_TOKENS=2048 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$BASELINE_REPO/python" \
"$BASELINE_REPO/.venv/bin/python" \
  python/infinilm/server/inference_server.py \
  --device nvidia \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --enable-paged-attn \
  --disable-prefix-caching \
  --num-blocks 768 \
  --block-size 256 \
  --max-batch-size 384 \
  --max-new-tokens 240 \
  --ignore-eos \
  --top-k 1 \
  --log-level ERROR
```

From a second shell, run one small connectivity check before the pressure run:

```bash
"$OPTIMIZED_REPO/.venv/bin/python" \
  "$OPTIMIZED_REPO/scripts/benchmark_serving_admission.py" \
  --model-path /models/TinyLlama-1.1B-Chat-v1.0 \
  --label baseline-smoke \
  --server-revision "$(git -C "$BASELINE_REPO" rev-parse HEAD)" \
  --wave1-requests 2 \
  --wave2-requests 1 \
  --output-tokens 8 \
  --warmup-requests 1
```

Then run the default pressure workload:

```bash
"$OPTIMIZED_REPO/.venv/bin/python" \
  "$OPTIMIZED_REPO/scripts/benchmark_serving_admission.py" \
  --model-path /models/TinyLlama-1.1B-Chat-v1.0 \
  --label baseline \
  --server-revision "$(git -C "$BASELINE_REPO" rev-parse HEAD)" \
  --output-json "$OPTIMIZED_REPO/results/baseline.json"
```

Stop the baseline server. Start the optimized server with the same model and
cache settings from the optimized checkout:

```bash
cd "$OPTIMIZED_REPO"

INFINILM_MAX_NUM_BATCHED_TOKENS=2048 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$OPTIMIZED_REPO/python" \
"$OPTIMIZED_REPO/.venv/bin/python" \
  python/infinilm/server/inference_server.py \
  --device nvidia \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --enable-paged-attn \
  --disable-prefix-caching \
  --num-blocks 768 \
  --block-size 256 \
  --max-batch-size 384 \
  --max-new-tokens 240 \
  --ignore-eos \
  --top-k 1 \
  --log-level ERROR
```

Run the optimized client measurement from the second shell:

```bash
"$OPTIMIZED_REPO/.venv/bin/python" \
  "$OPTIMIZED_REPO/scripts/benchmark_serving_admission.py" \
  --model-path /models/TinyLlama-1.1B-Chat-v1.0 \
  --label optimized \
  --server-revision "$(git -C "$OPTIMIZED_REPO" rev-parse HEAD)" \
  --output-json "$OPTIMIZED_REPO/results/optimized.json"
```

The optimized checkout may contain later documentation or benchmark commits;
the scheduler change itself was introduced by `81d4e55`. Verify that later
commits do not alter runtime Python code with `git diff 81d4e55 -- python`.

Alternate baseline and optimized runs at least five times rather than running
all samples for one revision together. Record GPU utilization and memory in a
third shell:

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
  --format=csv \
  --loop=1
```

The primary signal is wave 2 TTFT because it directly measures unnecessary
admission deferral. Also compare overall output tokens per second, requests per
second, TPOT, failures, and run-to-run variance. Run a second, non-pressure
profile with lower concurrency and more KV blocks to show the impact under a
normal GPU-bound workload.

## Limitations And Next Measurement

The current `25.03x` result is a scheduler control-plane microbenchmark, not an
end-to-end model throughput result. The staged benchmark has not yet been run
on NVIDIA hardware. `BlockManager.get_total_usable_blocks()` still scans used
blocks; tracking evictable capacity incrementally is a possible follow-up if
GPU-side profiling shows that it remains significant.
