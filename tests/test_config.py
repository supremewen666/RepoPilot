"""Tests for model configuration helpers."""

from __future__ import annotations

import os
import unittest

from repopilot.config import get_model_name, get_openai_base_url


class ConfigTestCase(unittest.TestCase):
    """Verify provider-agnostic model config behavior."""

    def test_model_name_defaults_from_model_identifier(self) -> None:
        old_model = os.environ.get("REPOPILOT_MODEL")
        old_name = os.environ.get("REPOPILOT_MODEL_NAME")
        os.environ["REPOPILOT_MODEL"] = "openai:gpt-4.1-mini"
        os.environ.pop("REPOPILOT_MODEL_NAME", None)
        try:
            self.assertEqual(get_model_name(), "gpt-4.1-mini")
        finally:
            if old_model is None:
                os.environ.pop("REPOPILOT_MODEL", None)
            else:
                os.environ["REPOPILOT_MODEL"] = old_model
            if old_name is None:
                os.environ.pop("REPOPILOT_MODEL_NAME", None)
            else:
                os.environ["REPOPILOT_MODEL_NAME"] = old_name

    def test_openai_base_url_can_be_set_for_compatible_provider(self) -> None:
        old_value = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
        try:
            self.assertEqual(get_openai_base_url(), "https://api.deepseek.com/v1")
        finally:
            if old_value is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = old_value


if __name__ == "__main__":
    unittest.main()
