import unittest

from infinilm.llm.request import FinishReason, InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler


def make_request(
    request_id: str,
    prompt_length: int,
    max_tokens: int,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_length)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class SchedulerAdmissionTest(unittest.TestCase):
    def test_decode_reservation_accounts_for_partial_last_block(self):
        scheduler = Scheduler(num_blocks=3, block_size=16)
        running_request = make_request("running", prompt_length=17, max_tokens=15)
        allocation = scheduler.cache_manager.allocate_slots(17)
        self.assertIsNotNone(allocation)
        running_request.block_table, running_request.slot_mapping = allocation
        running_request.append_generated_token_id(1)
        running_request.status = RequestStatus.RUNNING

        scheduler.complete_requests([running_request])

        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 0)
        candidate = make_request("candidate", prompt_length=1, max_tokens=15)
        self.assertTrue(scheduler.can_accept_request(candidate, 0))

    def test_running_decode_reservation_tracks_queue_lifecycle(self):
        scheduler = Scheduler(num_blocks=8, block_size=16)
        running_request = make_request("running", prompt_length=1, max_tokens=32)
        allocation = scheduler.cache_manager.allocate_slots(1)
        self.assertIsNotNone(allocation)
        running_request.block_table, running_request.slot_mapping = allocation
        running_request.append_generated_token_id(1)
        running_request.status = RequestStatus.RUNNING

        scheduler.complete_requests([running_request])
        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 2)

        output = scheduler.schedule()
        self.assertIsNotNone(output)
        self.assertEqual(output.scheduled_requests, [running_request])
        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 0)

        scheduler.complete_requests(output.scheduled_requests)
        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 2)

    def test_cache_stats_include_decode_reservation(self):
        scheduler = Scheduler(num_blocks=8, block_size=16)
        request = make_request("running", prompt_length=1, max_tokens=32)
        allocation = scheduler.cache_manager.allocate_slots(1)
        self.assertIsNotNone(allocation)
        request.block_table, request.slot_mapping = allocation
        request.status = RequestStatus.RUNNING

        scheduler.complete_requests([request])

        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 2)
        self.assertEqual(scheduler.get_cache_stats()["num_evictable_blocks"], 0)

    def test_canceled_running_request_releases_decode_reservation(self):
        scheduler = Scheduler(max_batch_size=1, num_blocks=8, block_size=16)
        requests = [
            make_request("canceled", prompt_length=1, max_tokens=32),
            make_request("next", prompt_length=1, max_tokens=32),
        ]
        for request in requests:
            allocation = scheduler.cache_manager.allocate_slots(1)
            self.assertIsNotNone(allocation)
            request.block_table, request.slot_mapping = allocation
            request.status = RequestStatus.RUNNING
            scheduler.complete_requests([request])

        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 4)
        requests[0].mark_canceled()

        output = scheduler.schedule()

        self.assertIsNotNone(output)
        self.assertEqual(output.scheduled_requests, [requests[1]])
        self.assertEqual(scheduler.get_cache_stats()["num_reserved_decode_blocks"], 0)


class SchedulerFairnessTest(unittest.TestCase):
    @staticmethod
    def finish_step(scheduler: Scheduler, output) -> None:
        for request in output.scheduled_requests:
            request.append_generated_token_id(1)
        scheduler.complete_requests(output.scheduled_requests)

    def test_cold_start_prefills_are_not_interleaved_with_decode(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=16,
            block_size=16,
            max_num_batched_tokens=16,
        )
        scheduler.add_request(make_request("first", prompt_length=16, max_tokens=4))
        scheduler.add_request(make_request("second", prompt_length=16, max_tokens=4))

        first_output = scheduler.schedule()
        self.assertTrue(first_output.is_prefill)
        self.finish_step(scheduler, first_output)

        second_output = scheduler.schedule()
        self.assertTrue(second_output.is_prefill)
        self.assertEqual(second_output.scheduled_requests[0].request_id, "second")

    def test_active_decode_limits_consecutive_prefill_batches(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=32,
            block_size=16,
            max_num_batched_tokens=16,
            max_consecutive_prefill_batches=1,
        )
        scheduler.add_request(make_request("active", prompt_length=16, max_tokens=8))

        prefill_output = scheduler.schedule()
        self.finish_step(scheduler, prefill_output)
        decode_output = scheduler.schedule()
        self.assertFalse(decode_output.is_prefill)
        self.finish_step(scheduler, decode_output)

        scheduler.add_request(make_request("new-1", prompt_length=16, max_tokens=4))
        scheduler.add_request(make_request("new-2", prompt_length=16, max_tokens=4))

        new_prefill_output = scheduler.schedule()
        self.assertTrue(new_prefill_output.is_prefill)
        self.assertEqual(new_prefill_output.scheduled_requests[0].request_id, "new-1")
        self.finish_step(scheduler, new_prefill_output)

        interleaved_decode_output = scheduler.schedule()
        self.assertFalse(interleaved_decode_output.is_prefill)
        self.assertEqual(
            interleaved_decode_output.scheduled_requests[0].request_id, "active"
        )

    def test_prefill_burst_limit_is_configurable(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=32,
            block_size=16,
            max_num_batched_tokens=16,
            max_consecutive_prefill_batches=2,
        )
        scheduler.add_request(make_request("active", prompt_length=16, max_tokens=8))
        prefill_output = scheduler.schedule()
        self.finish_step(scheduler, prefill_output)
        decode_output = scheduler.schedule()
        self.finish_step(scheduler, decode_output)

        for request_id in ("new-1", "new-2", "new-3"):
            scheduler.add_request(
                make_request(request_id, prompt_length=16, max_tokens=4)
            )

        first_prefill = scheduler.schedule()
        self.finish_step(scheduler, first_prefill)
        second_prefill = scheduler.schedule()
        self.finish_step(scheduler, second_prefill)

        self.assertTrue(first_prefill.is_prefill)
        self.assertTrue(second_prefill.is_prefill)
        decode_output = scheduler.schedule()
        self.assertFalse(decode_output.is_prefill)
        self.assertEqual(decode_output.scheduled_requests[0].request_id, "active")

    def test_finished_decode_request_does_not_force_interleaving(self):
        scheduler = Scheduler(
            max_batch_size=1,
            num_blocks=16,
            block_size=16,
            max_num_batched_tokens=16,
        )
        active = make_request("active", prompt_length=16, max_tokens=2)
        scheduler.add_request(active)

        prefill_output = scheduler.schedule()
        self.finish_step(scheduler, prefill_output)
        decode_output = scheduler.schedule()
        active.append_generated_token_id(1)
        active.mark_finished(FinishReason.LENGTH)
        scheduler.complete_requests(decode_output.scheduled_requests)

        scheduler.add_request(make_request("new-1", prompt_length=16, max_tokens=4))
        scheduler.add_request(make_request("new-2", prompt_length=16, max_tokens=4))

        first_output = scheduler.schedule()
        self.assertTrue(first_output.is_prefill)
        self.finish_step(scheduler, first_output)
        second_output = scheduler.schedule()
        self.assertTrue(second_output.is_prefill)


if __name__ == "__main__":
    unittest.main()
