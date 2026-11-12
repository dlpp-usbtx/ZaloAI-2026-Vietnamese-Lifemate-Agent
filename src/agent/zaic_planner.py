"""
src/agent/zaic_planner.py
Xử lý ReAct Loop, History ASR Tracing, Robust JSON Parsing và Retry logic.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple
from src.inference.engine import LLM_BATCH_RESPONSE_SCHEMA, LLM_RESPONSE_SCHEMA
from src.tools.catalog_manager import CatalogManager

class ZAICAgentPlanner:
    def __init__(
        self,
        llm_engine,
        catalog_manager: CatalogManager,
        asr_engine=None,
        max_llm_calls: int = 6,
        planning_mode: str = "react",
        max_available_tools: int | None = None,
    ):
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be greater than zero.")
        if planning_mode not in {"react", "batch"}:
            raise ValueError("planning_mode must be 'react' or 'batch'.")
        if max_available_tools is not None and max_available_tools < 1:
            raise ValueError("max_available_tools must be greater than zero.")
        self.llm_engine = llm_engine
        self.catalog_manager = catalog_manager
        self.asr_engine = asr_engine
        self.max_llm_calls = max_llm_calls
        self.planning_mode = planning_mode
        self.max_available_tools = max_available_tools
        self.last_schema_errors: List[Dict[str, Any]] = []
        self.last_performance_stats: Dict[str, Any] = {}

    def parse_llm_output_strict(self, output_text: str) -> Tuple[bool, Dict[str, Any], str]:
        try:
            start_idx = output_text.find("{")
            end_idx = output_text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return False, {}, "Không tìm thấy khối JSON {...} trong output."

            json_str = output_text[start_idx:end_idx]
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return False, {}, "Output JSON phải là một object."

            if "action" not in data:
                return False, {}, "Thiếu trường 'action' trong JSON."
            if data["action"] not in {"call_tool", "final_response"}:
                return False, {}, "Trường 'action' không hợp lệ."

            if data["action"] == "call_tool":
                tool_calls = data.get("tool_calls")
                if tool_calls is None:
                    tool_calls = [data.get("tool_call")]
                if not isinstance(tool_calls, list):
                    return False, {}, "Trường 'tool_calls' phải là danh sách."
                if not tool_calls or len(tool_calls) > 8:
                    return False, {}, "Cần có từ 1 đến 8 tool calls."
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict) or not isinstance(
                        tool_call.get("name"), str
                    ):
                        return False, {}, "Mỗi tool call cần có trường 'name'."
                    if not isinstance(tool_call.get("arguments", {}), dict):
                        return False, {}, "Trường 'arguments' phải là object."

            return True, data, "Valid"
        except Exception as e:
            return False, {}, f"JSON Parsing Exception: {str(e)}"

    def process_sample(self, item: Dict[str, Any], env, split_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
        planner_started = time.perf_counter()
        self.last_schema_errors = []
        llm_call_stats = []
        history_asr_seconds = 0.0
        tool_validation_seconds = 0.0
        tool_execution_seconds = 0.0
        history_asr_transcripts = []
        tool_execution_results = []
        available_tool_names = []
        turns_executed = 0
        context = item.get("context", {})
        user_profile = context.get("user_profile", {})
        current_location = context.get("location") or user_profile.get("home_address", "Không xác định")
        timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
        current_time = context.get("current_time", "")

        installed_apps = context.get("installed_apps", {})
        available_tools = self.catalog_manager.get_installed_tools(installed_apps)
        installed_tool_count = len(available_tools)
        if self.max_available_tools is not None:
            available_tools = self.catalog_manager.select_relevant_tools(
                available_tools,
                item.get("transcribed_text", ""),
                self.max_available_tools,
            )
        available_tool_names = [
            tool["name"] for tool in available_tools if isinstance(tool, dict) and "name" in tool
        ]

        response_schema = (
            LLM_BATCH_RESPONSE_SCHEMA
            if self.planning_mode == "batch"
            else LLM_RESPONSE_SCHEMA
        )
        if self.planning_mode == "batch":
            response_instructions = (
                "Chỉ trả JSON thuần, không markdown; thought viết ngắn. Lập kế hoạch "
                "đầy đủ ngay trong một lần: dùng action=call_tool với tool_calls là "
                "danh sách có thứ tự, tối đa 8 lệnh; gộp các lệnh độc lập và không "
                "lặp lệnh đọc nếu không cần. Nếu không cần tool, dùng final_response.\n"
                '{"thought":"...","action":"call_tool","tool_calls":'
                '[{"name":"...","arguments":{}},{"name":"...","arguments":{}}]}\n'
                '{"thought":"...","action":"final_response","content":"..."}'
            )
        else:
            response_instructions = (
                "Chỉ trả JSON thuần, không markdown; thought viết ngắn. Dùng đúng một dạng:\n"
                '{"thought":"...","action":"call_tool","tool_call":{"name":"...","arguments":{}}}\n'
                '{"thought":"...","action":"final_response","content":"..."}'
            )
        system_prompt = (
            f"Bạn là trợ lý cá nhân thông minh LifeMate.\n"
            f"Thời gian hiện tại: {current_time}\n"
            f"Múi giờ: {timezone}\n"
            f"Vị trí hiện tại: {current_location}\n"
            f"Thông tin người dùng: {json.dumps(user_profile, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"Danh sách Tools khả dụng:\n{json.dumps(available_tools, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"{response_instructions}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Chạy ASR cho các file audio lịch sử nếu thiếu text
        history = item.get("history", [])
        for history_index, h in enumerate(history):
            role = "user" if h.get("role") == "user" else "assistant"
            content = h.get("audio_text") or h.get("text") or h.get("content") or ""
            
            if not content and role == "user" and "audio" in h and self.asr_engine:
                audio_path = str(split_dir / h["audio"])
                history_asr_started = time.perf_counter()
                content = self.asr_engine.transcribe(audio_path)
                history_asr_seconds += time.perf_counter() - history_asr_started
                history_asr_transcripts.append(
                    {
                        "history_index": history_index,
                        "audio": h["audio"],
                        "transcript": content,
                    }
                )

            if content:
                messages.append({"role": role, "content": content})

        user_text = item.get("transcribed_text", "")
        messages.append(
            {
                "role": "user",
                "content": user_text,
                "_preserve_for_generation": True,
            }
        )

        max_turns = 1 if self.planning_mode == "batch" else 3
        llm_call_budget_reached = False
        followup_required = False

        for turn in range(max_turns):
            if len(llm_call_stats) >= self.max_llm_calls:
                llm_call_budget_reached = (
                    self.planning_mode == "react" and followup_required
                )
                break
            turns_executed = turn + 1
            retry_count = 0
            max_retries = 2
            turn_attempts = 0
            parsed_data = None
            is_valid = False

            while (
                retry_count < max_retries
                and len(llm_call_stats) < self.max_llm_calls
            ):
                call_started = time.perf_counter()
                raw_output = self.llm_engine.generate(
                    messages, response_schema=response_schema
                )
                call_seconds = time.perf_counter() - call_started
                turn_attempts += 1
                engine_stats = getattr(
                    self.llm_engine, "last_generation_stats", {}
                )
                call_stats = {
                    "turn": turn + 1,
                    "retry": retry_count + 1,
                    "call_seconds": call_seconds,
                    "engine": (
                        dict(engine_stats)
                        if isinstance(engine_stats, dict)
                        else {}
                    ),
                }
                llm_call_stats.append(call_stats)
                is_valid, parsed_data, err_msg = self.parse_llm_output_strict(raw_output)
                call_stats["json_valid"] = is_valid
                if not is_valid:
                    call_stats["format_error"] = err_msg
                    call_stats["output_preview"] = raw_output[:500]
                else:
                    call_stats["action"] = parsed_data.get("action")
                    if parsed_data.get("action") == "call_tool":
                        call_stats["tool_calls"] = (
                            parsed_data.get("tool_calls")
                            or [parsed_data.get("tool_call")]
                        )

                if is_valid:
                    break

                retry_count += 1
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"Lỗi định dạng JSON: {err_msg}. Hãy sửa và trả lại đúng định dạng JSON."})

            if not is_valid:
                llm_call_budget_reached = (
                    len(llm_call_stats) >= self.max_llm_calls
                )
                print(
                    f"Sample {item.get('id')}: không sinh được JSON hợp lệ "
                    f"sau {turn_attempts} lần thử; "
                    f"llm_call_budget={self.max_llm_calls}.",
                    flush=True,
                )
                break

            if parsed_data.get("action") == "final_response":
                break

            calls_to_execute = (
                parsed_data.get("tool_calls", [])
                if self.planning_mode == "batch"
                else [parsed_data.get("tool_call", {})]
            )
            followup_required = self.planning_mode == "react"
            for tool_call in calls_to_execute:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})

                validation_started = time.perf_counter()
                val_res = env.validate(tool_name, tool_args)
                tool_validation_seconds += time.perf_counter() - validation_started
                if not val_res.get("ok", True):
                    self.last_schema_errors.append(
                        {
                            "name": tool_name,
                            "arguments": tool_args,
                            "error": val_res.get("error"),
                        }
                    )
                    continue

                execution_started = time.perf_counter()
                exec_res = env.call(tool_name, tool_args)
                tool_execution_seconds += time.perf_counter() - execution_started
                tool_execution_results.append(
                    {
                        "name": tool_name,
                        "ok": bool(exec_res.get("ok", True)),
                        "done": exec_res.get("done"),
                        "result_keys": sorted(exec_res.keys()),
                    }
                )
                if self.planning_mode == "react":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(parsed_data, ensure_ascii=False),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Observation từ Tool {tool_name}]: "
                                f"{json.dumps(exec_res, ensure_ascii=False)}"
                            ),
                        }
                    )

        tool_calls = env.get_tool_calls()
        state_hash = env.get_state_hash()
        self.last_performance_stats = {
            "planner_seconds": time.perf_counter() - planner_started,
            "history_asr_seconds": history_asr_seconds,
            "llm_call_count": len(llm_call_stats),
            "llm_seconds": sum(
                call["call_seconds"] for call in llm_call_stats
            ),
            "tool_validation_seconds": tool_validation_seconds,
            "tool_execution_seconds": tool_execution_seconds,
            "turns_executed": turns_executed,
            "planning_mode": self.planning_mode,
            "max_available_tools": self.max_available_tools,
            "max_llm_calls": self.max_llm_calls,
            "llm_call_budget_reached": llm_call_budget_reached,
            "schema_error_count": len(self.last_schema_errors),
            "tool_call_count": len(tool_calls),
            "installed_tool_count": installed_tool_count,
            "prompt_tool_count": len(available_tools),
            "available_tool_names": available_tool_names,
            "history_asr_transcripts": history_asr_transcripts,
            "tool_execution_results": tool_execution_results,
            "llm_calls": llm_call_stats,
        }
        return tool_calls, state_hash