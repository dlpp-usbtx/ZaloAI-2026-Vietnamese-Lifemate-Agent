from src.agent.zaic_planner import ZAICAgentPlanner


class MockCatalogManager:
    def get_installed_tools(self, installed_apps):
        return []


class MockLLMEngine:
    def __init__(self):
        self.responses = [
            "not JSON",
            '{"action": "final_response", "content": "Done"}',
        ]
        self.last_generation_stats = {}

    def generate(self, messages):
        self.last_generation_stats = {
            "input_tokens": 12,
            "output_tokens": 4,
            "generation_seconds": 0.25,
        }
        return self.responses.pop(0)


class MockEnvironment:
    def get_tool_calls(self):
        return []

    def get_state_hash(self):
        return "state-hash"


def test_planner_records_per_call_retry_and_sample_metrics(tmp_path):
    llm_engine = MockLLMEngine()
    planner = ZAICAgentPlanner(llm_engine, MockCatalogManager())

    tool_calls, state_hash = planner.process_sample(
        {"id": "sample-1", "transcribed_text": "hello"},
        MockEnvironment(),
        tmp_path,
    )

    assert tool_calls == []
    assert state_hash == "state-hash"
    stats = planner.last_performance_stats
    assert stats["llm_call_count"] == 2
    assert stats["turns_executed"] == 1
    assert len(stats["llm_calls"]) == 2
    assert [
        (call["turn"], call["retry"]) for call in stats["llm_calls"]
    ] == [(1, 1), (1, 2)]
    assert all(call["engine"]["input_tokens"] == 12 for call in stats["llm_calls"])
    assert stats["planner_seconds"] >= 0
