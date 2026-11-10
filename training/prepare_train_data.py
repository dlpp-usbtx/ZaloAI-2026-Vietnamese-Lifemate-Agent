"""
training/prepare_train_data.py
Chuẩn hóa dữ liệu huấn luyện, xử lý đúng context location và lưu val_ids.json.
"""

from __future__ import annotations
import json
import random
import sys
from typing import Dict, Any, List
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import get_split_dir, BASE_DIR
from src.tools.catalog_manager import CatalogManager

def build_system_prompt(context: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> str:
    user_profile = context.get("user_profile", {})
    # Ưu tiên lấy vị trí hiện tại từ context.location, fallback về home_address trong user_profile
    current_location = context.get("location") or user_profile.get("home_address", "Không xác định")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    current_time = context.get("current_time", "")

    return (
        f"Bạn là trợ lý cá nhân thông minh LifeMate.\n"
        f"Thời gian hiện tại: {current_time}\n"
        f"Múi giờ: {timezone}\n"
        f"Vị trí hiện tại: {current_location}\n"
        f"Thông tin người dùng: {json.dumps(user_profile, ensure_ascii=False)}\n\n"
        f"Danh sách Tools khả dụng:\n{json.dumps(available_tools, ensure_ascii=False, indent=2)}\n\n"
        f"Hãy suy luận và phản hồi theo đúng định dạng JSON duy nhất:\n"
        f"{{\n"
        f'  "thought": "Suy luận...",\n'
        f'  "action": "call_tool" HOẶC "final_response",\n'
        f'  "tool_call": {{"name": "tên_tool", "arguments": {{...}}}},\n'
        f'  "content": "Câu trả lời nếu action = final_response"\n'
        f"}}\n"
    )

def convert_dataset_to_sft(val_ratio: float = 0.1, seed: int = 42):
    random.seed(seed)
    train_dir = get_split_dir("train")
    input_file = train_dir / "train.jsonl"
    catalog_file = train_dir / "train_tools_catalog.json"
    
    catalog_mgr = CatalogManager(catalog_file)
    raw_items = []

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            raw_items.append(json.loads(line))

    # Tách ID Validation cố định
    random.shuffle(raw_items)
    val_size = int(len(raw_items) * val_ratio)
    val_items = raw_items[:val_size]
    train_items = raw_items[val_size:]

    val_ids = [item["id"] for item in val_items]
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Lưu danh sách ID tập Validation để evaluator chỉ kiểm thử đúng các mẫu này
    with open(processed_dir / "val_ids.json", "w", encoding="utf-8") as f:
        json.dump(val_ids, f, ensure_ascii=False, indent=2)

    def process_item_list(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sft_records = []
        for item in items:
            context = item.get("context", {})
            user_text = item.get("audio_text", "")
            history = item.get("history", [])
            label = item.get("label", {})
            
            gold_calls = label.get("tool_calls", [])
            gold_results = label.get("tool_results", [])

            installed_apps = context.get("installed_apps", {})
            available_tools = catalog_mgr.get_installed_tools(installed_apps)
            system_prompt = build_system_prompt(context, available_tools)

            messages = [{"role": "system", "content": system_prompt}]

            for h in history:
                role = "user" if h.get("role") == "user" else "assistant"
                text_content = h.get("audio_text") or h.get("text") or h.get("content") or ""
                if text_content:
                    messages.append({"role": role, "content": text_content})

            messages.append({"role": "user", "content": user_text})

            if gold_calls:
                for idx, call in enumerate(gold_calls):
                    assistant_msg = {
                        "thought": f"Thực thi bước {idx + 1}.",
                        "action": "call_tool",
                        "tool_call": {"name": call.get("name"), "arguments": call.get("arguments", {})}
                    }
                    messages.append({"role": "assistant", "content": json.dumps(assistant_msg, ensure_ascii=False)})

                    if idx < len(gold_results):
                        tool_res = gold_results[idx]
                        obs_msg = {
                            "role": "user",
                            "content": f"[Observation từ Tool {call.get('name')}]: {json.dumps(tool_res, ensure_ascii=False)}"
                        }
                        messages.append(obs_msg)

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"thought": "Hoàn tất các bước.", "action": "final_response", "content": "Đã thực hiện xong."}, ensure_ascii=False)
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"thought": "Không cần gọi tool.", "action": "final_response", "content": "Dạ, em đã ghi nhận."}, ensure_ascii=False)
                })

            sft_records.append({"id": item["id"], "messages": messages})
        return sft_records

    train_data = process_item_list(train_items)
    val_data = process_item_list(val_items)

    with open(processed_dir / "sft_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(processed_dir / "sft_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã phân chia: {len(train_data)} Train samples & {len(val_data)} Validation samples.")
    print(f"📄 Đã lưu danh sách Val IDs tại: {processed_dir / 'val_ids.json'}")

if __name__ == "__main__":
    convert_dataset_to_sft()