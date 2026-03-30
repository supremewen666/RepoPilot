"""Tests for response normalization."""

import unittest

from repopilot.response_builder import build_final_response


class ResponseBuilderTestCase(unittest.TestCase):
    """Verify that raw orchestration output is normalized for the UI."""

    def test_build_final_response_converts_source_items(self) -> None:
        response = build_final_response(
            {
                "answer": "Grounded answer",
                "citations": [
                    {
                        "source_type": "doc",
                        "title": "Architecture",
                        "location": "/tmp/architecture.md",
                        "snippet": "RepoPilot uses a single agent.",
                    }
                ],
                "used_memory": ["User prefers concise answers"],
                "confidence": "high",
            }
        )

        self.assertEqual(response.answer, "Grounded answer")
        self.assertEqual(response.citations[0].label, "Architecture")
        self.assertEqual(response.citations[0].url_or_path, "/tmp/architecture.md")
        self.assertEqual(response.used_memory, ["User prefers concise answers"])
        self.assertEqual(response.confidence, "high")


if __name__ == "__main__":
    unittest.main()
