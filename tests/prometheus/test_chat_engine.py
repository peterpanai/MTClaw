"""chat_engine 单元测试。

测试范围：
- is_chat_intent 闲聊意图识别（5条规则 + 误判保护）
- detect_mood 情绪检测
- build_prompt prompt 构造
"""
import sys
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR / "chat"))


@pytest.fixture
def engine():
    if "chat_engine" in sys.modules:
        del sys.modules["chat_engine"]
    import chat_engine
    return chat_engine


class TestIsChatIntent:
    """测试闲聊意图识别。"""

    @pytest.mark.parametrize("message,expected", [
        # 正面：纯闲聊
        ("你好", True),
        ("嗨", True),
        ("早上好", True),
        ("晚安", True),
        ("讲个笑话", True),
        ("好无聊啊", True),
        ("今天好累", True),
        ("烦死了", True),
        ("哈哈", True),
        ("好的", True),
        ("谢谢", True),
        # 负面：非闲聊（complex 消息）
        ("帮我写一份周报，用 Markdown 格式", False),  # 写作意图 + 太长
        ("找一下关于 GPU 算力的笔记", False),  # RAG 检索
        ("明天下午3点开产品评审会", False),  # 日程
        ("请分析这段代码的性能瓶颈", False),  # 分析类
        ("对比一下 A100 和 H100 的算力", False),  # 对比分析
        (".py 文件怎么运行", False),  # 文件扩展名
        ("/home/pmc/report.md", False),  # 文件路径
        # 边界
        ("", False),  # 空字符串
        ("x" * 250, False),  # 超长消息
    ])
    def test_intent_detection(self, engine, message, expected):
        assert engine.is_chat_intent(message) == expected

    def test_no_file_extension(self, engine):
        assert engine.is_chat_intent("config.py") == False

    def test_long_message_rejected(self, engine):
        assert engine.is_chat_intent("你好" * 101) == False

    def test_complex_indicators_block_chat(self, engine):
        """包含技术关键词的消息不应走闲聊。"""
        for word in ["代码", "函数", "数据库", "部署", "架构", "接口"]:
            assert engine.is_chat_intent(f"你好，帮我看看{word}") == False


class TestDetectMood:
    """测试情绪检测。"""

    @pytest.mark.parametrize("message,expected_mood", [
        ("好难过啊", "comfort"),
        ("烦死了", "comfort"),
        ("好累", "comfort"),
        ("开心", "casual"),
        ("讲个笑话", "humor"),
        ("有趣的故事", "humor"),  # "有趣"匹配 ENTERTAINMENT 先于 curious
        ("你好", "casual"),  # 无明确情绪 -> casual
    ])
    def test_mood_detection(self, engine, message, expected_mood):
        assert engine.detect_mood(message) == expected_mood


class TestBuildPrompt:
    """测试 prompt 构造。"""

    def test_prompt_contains_mood_strategy(self, engine):
        prompt = engine.build_prompt("humor", "")
        assert "幽默" in prompt or "笑话" in prompt or "humor" in prompt.lower()

    def test_prompt_contains_memory(self, engine):
        prompt = engine.build_prompt("casual", "- 昵称: 小明\n- 兴趣: 编程")
        assert "小明" in prompt

    def test_prompt_no_memory(self, engine):
        prompt = engine.build_prompt("casual", "")
        assert len(prompt) > 0  # 不应崩溃


class TestChatLightFunction:
    """测试 chat_light 主函数。"""

    def test_chat_light_no_api(self, engine, monkeypatch):
        """没有配置路由模型时应返回 fallback 回复。"""
        monkeypatch.setattr(engine, "ROUTING_URL", "")
        monkeypatch.setattr(engine, "ROUTING_MODEL", "")
        result = engine.chat_light({"mood": "casual", "_user_message": "你好"})
        assert result["status"] == "ok"
        assert len(result["reply"]) > 0
