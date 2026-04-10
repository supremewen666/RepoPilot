"""Static checks for the teaching-first module boundaries."""

from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ArchitectureBoundaryTestCase(unittest.TestCase):
    """Verify canonical modules follow the new boundary rules."""

    def test_canonical_rag_modules_do_not_import_service_layer(self) -> None:
        canonical_paths = [
            PROJECT_ROOT / "repopilot" / "rag" / "orchestrator.py",
            *sorted((PROJECT_ROOT / "repopilot" / "rag" / "indexing").glob("*.py")),
            *sorted((PROJECT_ROOT / "repopilot" / "rag" / "knowledge").glob("*.py")),
            *sorted((PROJECT_ROOT / "repopilot" / "rag" / "retrieval").glob("*.py")),
            *sorted((PROJECT_ROOT / "repopilot" / "rag" / "storage").rglob("*.py")),
            PROJECT_ROOT / "repopilot" / "rag" / "types.py",
        ]
        for path in canonical_paths:
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("repopilot.service", content, path.as_posix())

    def test_service_entrypoints_do_not_import_internal_rag_layers(self) -> None:
        service_paths = [
            *sorted((PROJECT_ROOT / "repopilot" / "service" / "api").glob("*.py")),
            *sorted((PROJECT_ROOT / "repopilot" / "service" / "agent").glob("*.py")),
            PROJECT_ROOT / "repopilot" / "service" / "tasks" / "manager.py",
        ]
        forbidden_markers = (
            "repopilot.rag.indexing",
            "repopilot.rag.knowledge",
            "repopilot.rag.retrieval",
            "repopilot.rag.storage.",
        )
        for path in service_paths:
            content = path.read_text(encoding="utf-8")
            for marker in forbidden_markers:
                self.assertNotIn(marker, content, f"{path.as_posix()} imports {marker}")
