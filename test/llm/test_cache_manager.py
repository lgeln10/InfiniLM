import unittest

from infinilm.llm.cache_manager import BlockManager
from infinilm.llm.prefix_cache import hash_block_tokens


class BlockManagerUsableCapacityTest(unittest.TestCase):
    def test_released_blocks_become_usable(self):
        manager = BlockManager(num_blocks=4, block_size=4)
        block_table, _ = manager.allocate_slots(8)

        self.assertEqual(manager.get_total_usable_blocks(), 2)
        manager.free_blocks(block_table)

        self.assertEqual(manager.get_num_free_blocks(), 2)
        self.assertEqual(manager.get_total_usable_blocks(), 4)
        self.assertEqual(manager.evictable_block_ids, set(block_table))

    def test_prefix_hit_pins_an_evictable_block(self):
        manager = BlockManager(num_blocks=2, block_size=4)
        block_table, _ = manager.allocate_slots(4)
        block_hash = hash_block_tokens(range(4))
        manager.publish_computed_blocks(block_table, [block_hash], 0, 4)
        manager.free_blocks(block_table)

        cached_blocks, cached_tokens = manager.get_computed_blocks([block_hash], 4)

        self.assertEqual(cached_blocks, block_table)
        self.assertEqual(cached_tokens, 4)
        self.assertEqual(manager.get_total_usable_blocks(), 1)
        self.assertFalse(manager.evictable_block_ids)

        manager.free_blocks(cached_blocks)
        self.assertEqual(manager.get_total_usable_blocks(), 2)

    def test_allocation_evicts_only_needed_unreferenced_blocks(self):
        manager = BlockManager(num_blocks=3, block_size=4)
        old_blocks, _ = manager.allocate_slots(12)
        manager.free_blocks(old_blocks)

        new_blocks, _ = manager.allocate_slots(4)

        self.assertEqual(len(new_blocks), 1)
        self.assertEqual(manager.get_num_free_blocks(), 0)
        self.assertEqual(len(manager.evictable_block_ids), 2)
        self.assertEqual(manager.get_total_usable_blocks(), 2)

    def test_truncation_preserves_usable_capacity_counter(self):
        manager = BlockManager(num_blocks=3, block_size=4)
        block_table, _ = manager.allocate_slots(8)

        retained = manager.truncate_blocks(block_table, keep_num_tokens=4)

        self.assertEqual(len(retained), 1)
        self.assertEqual(manager.get_total_usable_blocks(), 2)
        manager.free_blocks(retained)
        self.assertEqual(manager.get_total_usable_blocks(), 3)


if __name__ == "__main__":
    unittest.main()
