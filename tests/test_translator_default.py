import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_planning_analysis.utils import text_to_pddl


def test_text_to_plan_with_llm_uses_openrouter_default(monkeypatch):
    captured = {}

    class FakeChatCompletions:
        def create(self, model, messages):
            captured["model"] = model
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="[PDDL PLAN]\n(move a b)\n[PDDL PLAN END]"
                        )
                    )
                ]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(text_to_pddl, "OpenAI", lambda: FakeClient())

    plan, raw_translation = text_to_pddl.text_to_plan_with_llm(
        "Sample plan response",
        {"domain_name": "test_domain"},
        {"instance_id": 1},
    )

    assert captured["model"] == "openai/gpt-4o"
    assert plan == "(move a b)"
    assert raw_translation == "[PDDL PLAN]\n(move a b)\n[PDDL PLAN END]"
