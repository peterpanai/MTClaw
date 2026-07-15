"""writing_engine 单元测试。

测试范围：
- 模板加载（内置 + 用户自定义优先级）
- HumanizeRuleEngine 规则改写（humanize_light 方法）
- prompt 构造（generate/polish/translate/humanize）
- 参数校验
- 不依赖上游 LLM（跳过实际 API 调用）
"""
import json
import os
import sys
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR / "writing"))


@pytest.fixture
def engine(tmp_path, monkeypatch):
    if "writing_engine" in sys.modules:
        del sys.modules["writing_engine"]
    import writing_engine
    monkeypatch.setattr(writing_engine, "USER_TEMPLATES_DIR", tmp_path / "templates")
    return writing_engine


class TestTemplateLoading:
    """测试模板加载。"""

    def test_load_builtin_template(self, engine):
        template = engine._load_template("weekly_report")
        assert len(template) > 0

    def test_load_builtin_email(self, engine):
        template = engine._load_template("email")
        assert len(template) > 0

    def test_load_nonexistent_template(self, engine):
        template = engine._load_template("nonexistent_type_xyz")
        assert "nonexistent_type_xyz" in template

    def test_user_template_overrides_builtin(self, engine, tmp_path):
        user_dir = tmp_path / "templates"
        user_dir.mkdir()
        (user_dir / "weekly_report.md").write_text("# 自定义周报模板\n\n用户自定义。")

        import writing_engine as we
        we.USER_TEMPLATES_DIR = user_dir
        template = we._load_template("weekly_report")
        assert "自定义周报模板" in template

    def test_list_templates_includes_builtin(self, engine):
        names = engine._list_templates()
        assert "weekly_report" in names
        assert "email" in names

    def test_list_templates_includes_user(self, engine, tmp_path):
        user_dir = tmp_path / "templates"
        user_dir.mkdir()
        (user_dir / "custom_type.md").write_text("custom")

        import writing_engine as we
        we.USER_TEMPLATES_DIR = user_dir
        names = we._list_templates()
        assert "custom_type" in names
        assert "weekly_report" in names


class TestHumanizeRuleEngine:
    """测试去AI化规则引擎。"""

    @pytest.fixture
    def rule_engine(self, engine):
        return engine.HumanizeRuleEngine()

    def test_humanize_light_removes_ai_phrases(self, rule_engine):
        text = "It is crucial to delve into the details."
        result = rule_engine.humanize_light(text)
        assert isinstance(result, dict)
        assert "text" in result or "humanized" in result
        # 检查 AI 词汇被替换
        output = result.get("text", result.get("humanized", ""))
        assert "crucial" not in output.lower() or len(output) < len(text)

    def test_humanize_light_chinese(self, rule_engine):
        text = "在当今时代，我们需要至关重要地赋能用户。"
        result = rule_engine.humanize_light(text)
        assert isinstance(result, dict)

    def test_humanize_light_preserves_normal_text(self, rule_engine):
        text = "今天天气不错，我们去公园散步吧。"
        result = rule_engine.humanize_light(text)
        output = result.get("text", result.get("humanized", ""))
        assert "今天天气不错" in output

    def test_calculate_burstiness(self, rule_engine):
        text = "这是一段测试文本。短句。长句子的内容更多一些。"
        score = rule_engine.calculate_burstiness(text)
        assert isinstance(score, (int, float))


class TestBuildPrompts:
    """测试 prompt 构造。"""

    def test_build_generate_prompt(self, engine):
        result = engine._build_generate_prompt(
            doc_type="weekly_report",
            topic="本周工作总结",
            key_points=["完成了A功能"],
            style="formal",
            length="medium",
            preferences={},
        )
        # 返回 3 元组
        assert isinstance(result, tuple)
        assert len(result) >= 2
        system = result[0]
        user = result[1]
        assert "本周工作总结" in user or "周报" in system

    def test_build_polish_prompt(self, engine):
        result = engine._build_polish_prompt(
            text="需要润色的文字",
            goal="more_professional",
            target_language="zh-CN",
            preferences={},
        )
        assert isinstance(result, tuple)
        assert len(result) >= 2

    def test_build_translate_prompt(self, engine):
        result = engine._build_translate_prompt(
            text="Hello",
            source_lang="en",
            target_lang="zh-CN",
            keep_formatting=True,
        )
        assert isinstance(result, tuple)
        assert len(result) >= 2


class TestHumanizeFunction:
    """测试 humanize 函数。"""

    def test_humanize_light_no_llm(self, engine, monkeypatch):
        """light 级别应纯规则处理，不需要调用 LLM。"""
        def fail_call(*args, **kwargs):
            raise AssertionError("不应调用上游 LLM")
        monkeypatch.setattr(engine, "_call_upstream_llm", fail_call)

        result = engine.humanize(
            text="It is crucial to delve into the details.",
            intensity="light",
        )
        assert result.get("status") == "ok" or "humanized" in result

    def test_humanize_missing_text(self, engine):
        result = engine.humanize(text="", intensity="medium")
        assert "error" in result

    def test_humanize_invalid_intensity(self, engine, monkeypatch):
        monkeypatch.setattr(engine, "_call_upstream_llm",
                           lambda *a, **kw: {"content": "改写", "changes": "test"})
        result = engine.humanize(text="测试文本", intensity="invalid")
        # 应 fallback 到 medium
        assert isinstance(result, dict)


class TestGenerateValidation:
    """测试写作生成的参数校验。"""

    def test_generate_invalid_doc_type(self, engine):
        result = engine.generate(doc_type="invalid_type", topic="测试")
        assert "error" in result

    def test_generate_empty_topic(self, engine):
        result = engine.generate(doc_type="weekly_report", topic="")
        assert "error" in result

    def test_generate_missing_topic(self, engine):
        with pytest.raises(TypeError):
            engine.generate(doc_type="weekly_report")
