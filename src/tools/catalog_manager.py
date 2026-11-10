"""
src/tools/catalog_manager.py
Lọc chính xác danh sách tools được cài đặt cho từng sample, phát hiện lỗi catalog file.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

SYS_ASK_MISSING_INFO_SCHEMA = {
    "name": "sys.ask_missing_info",
    "description": "Yêu cầu người dùng cung cấp thêm thông tin còn thiếu khi không đủ tham số bắt buộc.",
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "description": "Tên tool đang bị thiếu thông tin"},
            "params": {"type": "array", "items": {"type": "string"}, "description": "Danh sách các tham số còn thiếu"}
        },
        "required": ["tool_name", "params"]
    }
}

class CatalogManager:
    def __init__(self, catalog_path: Path):
        self.all_tools: Dict[str, Dict[str, Any]] = {}
        self._load_catalog(catalog_path)

    def _load_catalog(self, catalog_path: Path):
        if not catalog_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy Catalog File tại: {catalog_path}")
            
        try:
            with open(catalog_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Lỗi khi đọc Catalog File {catalog_path}: {str(e)}")

        for app in data.get("apps", []):
            for tool in app.get("tools", []):
                tool_name = tool.get("name")
                if tool_name:
                    self.all_tools[tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "kind": tool.get("kind", "")
                    }

    def get_installed_tools(self, installed_apps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Lấy chính xác hợp danh sách tools được cài cho mẫu dựa vào installed_apps[app_id]['tools'].
        Thêm sys.ask_missing_info duy nhất 1 lần và loại bỏ trùng lặp.
        """
        installed_tool_names = set()
        
        for app_id, app_info in installed_apps.items():
            if isinstance(app_info, dict) and "tools" in app_info:
                for tool_obj in app_info["tools"]:
                    if isinstance(tool_obj, dict) and "name" in tool_obj:
                        installed_tool_names.add(tool_obj["name"])
                    elif isinstance(tool_obj, str):
                        installed_tool_names.add(tool_obj)

        tools_list = []
        # Thêm sys.ask_missing_info đầu tiên
        tools_list.append(SYS_ASK_MISSING_INFO_SCHEMA)
        
        for name in sorted(installed_tool_names):
            if name == "sys.ask_missing_info":
                continue
            if name in self.all_tools:
                tools_list.append(self.all_tools[name])

        return tools_list