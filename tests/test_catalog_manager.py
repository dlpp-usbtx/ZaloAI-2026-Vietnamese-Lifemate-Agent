from src.tools.catalog_manager import CatalogManager


def test_relevant_tool_selection_matches_vietnamese_and_keeps_system_tool():
    manager = CatalogManager.__new__(CatalogManager)
    tools = [
        {"name": "sys.ask_missing_info", "description": "Hỏi thêm thông tin"},
        {"name": "tasks.add", "description": "Thêm việc cần làm"},
        {"name": "medication.add", "description": "Thêm lịch uống thuốc"},
        {"name": "reminders.create", "description": "Tạo nhắc việc"},
    ]

    selected = manager.select_relevant_tools(
        tools,
        "Nhắc uống vitamin D sáng nay",
        max_tools=2,
    )

    assert [tool["name"] for tool in selected] == [
        "sys.ask_missing_info",
        "medication.add",
        "reminders.create",
    ]


def test_relevant_tool_selection_falls_back_when_no_terms_match():
    manager = CatalogManager.__new__(CatalogManager)
    tools = [
        {"name": "sys.ask_missing_info", "description": "Hỏi thêm thông tin"},
        {"name": "weather.get_current", "description": "Thời tiết hiện tại"},
        {"name": "calendar.list_events", "description": "Sự kiện lịch"},
    ]

    assert manager.select_relevant_tools(
        tools,
        "qzxv",
        max_tools=1,
    ) == tools
