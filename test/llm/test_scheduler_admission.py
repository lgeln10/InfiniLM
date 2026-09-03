import unittest

from infinilm.llm.request import InferenceRequest, RequestStatus
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


if __name__ == "__main__":
    unittest.main()
