"""
src/agent/zaic_planner.py
Xử lý ReAct Loop, History ASR Tracing, Robust JSON Parsing và Retry logic.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.tools.catalog_manager import CatalogManager

class ZAICAgentPlanner:
    def __init__(self, llm_engine, catalog_manager: CatalogManager, asr_engine=None):
        self.llm_engine = llm_engine
        self.catalog_manager = catalog_manager
        self.asr_engine = asr_engine
        self.last_schema_errors: List[Dict[str, Any]] = []

    def parse_llm_output_strict(self, output_text: str) -> Tuple[bool, Dict[str, Any], str]:
        try:
            start_idx = output_text.find("{")
            end_idx = output_text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return False, {}, "Không tìm thấy khối JSON {...} trong output."

            json_str = output_text[start_idx:end_idx]
            data = json.loads(json_str)

            if "action" not in data:
                return False, {}, "Thiếu trường 'action' trong JSON."

            if data["action"] == "call_tool":
                if "tool_call" not in data or "name" not in data["tool_call"]:
                    return False, {}, "Thiếu trường 'tool_call.name' khi action = 'call_tool'."

            return True, data, "Valid"
        except Exception as e:
            return False, {}, f"JSON Parsing Exception: {str(e)}"

    def process_sample(self, item: Dict[str, Any], env, split_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
        self.last_schema_errors = []
        context = item.get("context", {})
        user_profile = context.get("user_profile", {})
        current_location = context.get("location") or user_profile.get("home_address", "Không xác định")
        timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
        current_time = context.get("current_time", "")

        installed_apps = context.get("installed_apps", {})
        available_tools = self.catalog_manager.get_installed_tools(installed_apps)

        system_prompt = (
            f"Bạn là trợ lý cá nhân thông minh LifeMate.\n"
            f"Thời gian hiện tại: {current_time}\n"
            f"Múi giờ: {timezone}\n"
            f"Vị trí hiện tại: {current_location}\n"
            f"Thông tin người dùng: {json.dumps(user_profile, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"Danh sách Tools khả dụng:\n{json.dumps(available_tools, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"Hãy suy luận và phản hồi theo đúng định dạng JSON duy nhất:\n"
            f"{{\n"
            f'  "thought": "Suy luận...",\n'
            f'  "action": "call_tool" HOẶC "final_response",\n'
            f'  "tool_call": {{"name": "tên_tool", "arguments": {{...}}}},\n'
            f'  "content": "Câu trả lời nếu action = final_response"\n'
            f"}}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Chạy ASR cho các file audio lịch sử nếu thiếu text
        history = item.get("history", [])
        for h in history:
            role = "user" if h.get("role") == "user" else "assistant"
            content = h.get("audio_text") or h.get("text") or h.get("content") or ""
            
            if not content and role == "user" and "audio" in h and self.asr_engine:
                audio_path = str(split_dir / h["audio"])
                content = self.asr_engine.transcribe(audio_path)

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

        max_turns = 3
        execution_failed = False

        for turn in range(max_turns):
            retry_count = 0
            max_retries = 2
            parsed_data = None
            is_valid = False

            while retry_count < max_retries:
                raw_output = self.llm_engine.generate(messages)
                is_valid, parsed_data, err_msg = self.parse_llm_output_strict(raw_output)

                if is_valid:
                    break

                retry_count += 1
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"Lỗi định dạng JSON: {err_msg}. Hãy sửa và trả lại đúng định dạng JSON."})

            if not is_valid:
                print(f"❌ Sample {item.get('id')}: Thất bại khi sinh JSON hợp lệ sau {max_retries} lần thử.")
                execution_failed = True
                break

            if parsed_data.get("action") == "final_response":
                break

            tool_call = parsed_data.get("tool_call", {})
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            val_res = env.validate(tool_name, tool_args)
            if not val_res.get("ok", True):
                self.last_schema_errors.append(
                    {
                        "name": tool_name,
                        "arguments": tool_args,
                        "error": val_res.get("error"),
                    }
                )
                messages.append({"role": "assistant", "content": json.dumps(parsed_data, ensure_ascii=False)})
                messages.append({"role": "user", "content": f"Lỗi tham số Tool: {val_res.get('error')}. Hãy sửa lại arguments."})
                continue

            exec_res = env.call(tool_name, tool_args)
            messages.append({"role": "assistant", "content": json.dumps(parsed_data, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"[Observation từ Tool {tool_name}]: {json.dumps(exec_res, ensure_ascii=False)}"})

        return env.get_tool_calls(), env.get_state_hash()