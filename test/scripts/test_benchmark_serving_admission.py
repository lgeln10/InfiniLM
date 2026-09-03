import asyncio
import unittest
from types import SimpleNamespace

from scripts.benchmark_serving_admission import (
    FirstTokenBarrier,
    RequestResult,
    issue_request,
    percentile,
    predict_kv_pressure,
    run_staged_benchmark,
    summarize_results,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": text.split()}


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        self.iterator = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCompletions:
    async def create(self, **kwargs):
        del kwargs
        return FakeStream(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="one"),
                            finish_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None),
                            finish_reason="length",
                        )
                    ]
                ),
            ]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FailingCompletions:
    async def create(self, **kwargs):
        del kwargs
        raise RuntimeError("server unavailable")


class FailingClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FailingCompletions())


class BenchmarkServingAdmissionTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(percentile(values, 0.5), 2.5)
        self.assertAlmostEqual(percentile(values, 0.95), 3.85)

    def test_pressure_prediction_matches_target_workload(self):
        prediction = predict_kv_pressure(
            prompt_tokens=257,
            output_tokens=240,
            block_size=256,
            num_blocks=768,
            wave1_requests=256,
            wave2_requests=128,
        )

        self.assertEqual(prediction["prompt_blocks_per_request"], 2)
        self.assertEqual(prediction["target_blocks_per_request"], 2)
        self.assertEqual(
            prediction["legacy_running_reservation_per_request"],
            1,
        )
        self.assertEqual(prediction["exact_running_reservation_per_request"], 0)
        self.assertEqual(prediction["false_reserved_blocks_after_wave1"], 256)
        self.assertEqual(prediction["predicted_wave2_admissions_legacy"], 0)
        self.assertEqual(prediction["predicted_wave2_admissions_exact"], 128)

    def test_summary_excludes_failed_requests(self):
        results = [
            RequestResult(
                wave="wave1",
                request_index=0,
                started_at=10.0,
                first_token_at=10.1,
                finished_at=11.0,
                finish_reason="length",
                output_tokens=10,
                content_chunks=10,
            ),
            RequestResult(
                wave="wave1",
                request_index=1,
                started_at=10.0,
                first_token_at=None,
                finished_at=12.0,
                finish_reason=None,
                output_tokens=0,
                content_chunks=0,
                error="failed",
            ),
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["attempted_requests"], 2)
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["output_tokens"], 10)
        self.assertAlmostEqual(summary["duration_s"], 2.0)
        self.assertAlmostEqual(summary["requests_per_second"], 0.5)
        self.assertAlmostEqual(summary["output_tokens_per_second"], 5.0)
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 100.0)
        self.assertAlmostEqual(summary["tpot_ms"]["p50"], 100.0)


class BenchmarkServingAdmissionAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_issue_request_marks_barrier_and_counts_length_finish(self):
        barrier = FirstTokenBarrier(target=1, total=1)

        result = await issue_request(
            client=FakeClient(),
            tokenizer=FakeTokenizer(),
            model="test-model",
            prompt_content="prompt",
            output_tokens=4,
            wave="wave1",
            request_index=0,
            first_token_barrier=barrier,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.output_tokens, 4)
        self.assertEqual(result.content_chunks, 1)
        self.assertEqual(barrier.count, 1)
        self.assertTrue(barrier.event.is_set())

    async def test_staged_benchmark_fails_when_barrier_is_impossible(self):
        args = SimpleNamespace(
            wave1_requests=2,
            wave2_requests=1,
            wave2_start_after=2,
            output_tokens=4,
            model="test-model",
            barrier_timeout=60.0,
        )

        with self.assertRaisesRegex(RuntimeError, "cannot reach"):
            await asyncio.wait_for(
                run_staged_benchmark(
                    client=FailingClient(),
                    tokenizer=FakeTokenizer(),
                    args=args,
                    prompt_content="prompt",
                ),
                timeout=1.0,
            )


if __name__ == "__main__":
    unittest.main()
