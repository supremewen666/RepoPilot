"""Tests for memory heuristics and fallback storage."""

from __future__ import annotations

import os
import tempfile
import unittest

from repopilot.memory.store import get_relevant_memories, save_memory_if_needed, should_persist_memory


class MemoryStoreTestCase(unittest.TestCase):
    """Exercise the fallback memory path without requiring mem0."""

    def test_should_persist_memory_filters_transient_content(self) -> None:
        self.assertTrue(should_persist_memory("I prefer Chinese responses.", "Okay."))
        self.assertFalse(should_persist_memory("What is LangChain?", "It is a framework."))

    def test_memory_fallback_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            old_value = os.environ.get("REPOPILOT_MEMORY_STORE_PATH")
            os.environ["REPOPILOT_MEMORY_STORE_PATH"] = os.path.join(tmp_dir, "memory.json")
            try:
                save_memory_if_needed(
                    user_id="u-1",
                    user_query="I prefer Chinese responses and I am currently working on PR #12.",
                    assistant_answer="I will remember that preference.",
                )
                memories = get_relevant_memories(user_id="u-1", query="What am I working on?")
            finally:
                if old_value is None:
                    os.environ.pop("REPOPILOT_MEMORY_STORE_PATH", None)
                else:
                    os.environ["REPOPILOT_MEMORY_STORE_PATH"] = old_value

        self.assertTrue(memories)
        self.assertTrue("Chinese" in memories[0] or "PR #12" in memories[0])


if __name__ == "__main__":
    unittest.main()
