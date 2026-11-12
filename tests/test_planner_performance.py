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


class ToolCallingLLMEngine:
    def __init__(self):
        self.call_count = 0
        self.last_generation_stats = {}

    def generate(self, messages):
        self.call_count += 1
        self.last_generation_stats = {
            "input_tokens": 10,
            "output_tokens": 8,
            "generation_seconds": 0.5,
        }
        return (
            '{"thought":"Thực thi bước 1.","action":"call_tool",'
            '"tool_call":{"name":"calendar.create","arguments":{}}}'
        )


class ToolEnvironment(MockEnvironment):
    def __init__(self):
        self.executed_calls = []

    def validate(self, tool_name, tool_args):
        return {"ok": True}

    def call(self, tool_name, tool_args):
        self.executed_calls.append({"name": tool_name, "arguments": tool_args})
        return {"ok": True}

    def get_tool_calls(self):
        return self.executed_calls


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


def test_planner_respects_total_llm_call_budget(tmp_path):
    llm_engine = ToolCallingLLMEngine()
    planner = ZAICAgentPlanner(
        llm_engine, MockCatalogManager(), max_llm_calls=2
    )
    environment = ToolEnvironment()

    tool_calls, state_hash = planner.process_sample(
        {"id": "sample-2", "transcribed_text": "create two events"},
        environment,
        tmp_path,
    )

    assert len(tool_calls) == 2
    assert state_hash == "state-hash"
    assert llm_engine.call_count == 2
    assert planner.last_performance_stats["llm_call_budget_reached"] is True
    assert planner.last_performance_stats["max_llm_calls"] == 2
