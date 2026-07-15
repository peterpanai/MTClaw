#!/usr/bin/env python3
"""Writing Engine - document generation, polishing, translation & humanization.

Architecture:
  - Layer 1: Global system prompt (role + rules, shared across all doc types)
  - Layer 2: Document-type template (loaded from templates/*.md)
  - Layer 3: Dynamic injection (user preferences via memory_recall + task params)

Tools exposed (selected via argv[1]):
  - writing_generate   : Generate documents from topic + key_points
  - writing_polish     : Polish/refine existing text
  - writing_translate  : Translate text between languages
  - writing_humanize   : Remove AI writing patterns (3 intensity levels)

Usage:
  echo '{"doc_type":"weekly_report","topic":"...","key_points":[...]}' | \
    python3 writing_engine.py writing_generate

Input is JSON on stdin; output is JSON on stdout (compatible with MTClaw FR).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
import urllib.request
import urllib.error

logger = logging.getLogger("writing_engine")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = SCRIPT_DIR / "templates"
# 用户自定义模板目录（优先级高于内置模板，用户可在此添加/覆盖模板）
USER_TEMPLATES_DIR = Path(os.environ.get(
    "PROMETHEUS_TEMPLATES_DIR",
    os.path.expanduser("~/.function-router/prometheus/templates"),
))

# Upstream LLM configuration (env vars or config file)
UPSTREAM_URL = os.environ.get("WRITING_UPSTREAM_URL", os.environ.get("UPSTREAM_URL", ""))
UPSTREAM_MODEL = os.environ.get("WRITING_UPSTREAM_MODEL", os.environ.get("UPSTREAM_MODEL", ""))
UPSTREAM_KEY = os.environ.get("WRITING_UPSTREAM_KEY", os.environ.get("UPSTREAM_API_KEY", os.environ.get("API_KEY", "")))

# Memory engine path (for preference recall)
MEMORY_ENGINE = os.environ.get("MEMORY_ENGINE_PATH", str(Path(__file__).resolve().parent.parent / "memory" / "memory_engine.py"))

# ---------------------------------------------------------------------------
# Valid parameter values
# ---------------------------------------------------------------------------

VALID_DOC_TYPES = {
    "weekly_report", "email", "tech_doc", "meeting_minutes",
    "article", "essay", "ppt_outline",
}

VALID_STYLES = {"formal", "casual", "technical", "academic"}
VALID_LENGTHS = {"short", "medium", "long"}
VALID_GOALS = {"more_professional", "more_concise", "more_friendly", "fix_grammar", "more_technical"}
VALID_INTENSITIES = {"light", "medium", "heavy"}

# Temperature per doc type
TEMP_MAP = {
    "weekly_report": 0.3, "email": 0.4, "tech_doc": 0.2,
    "meeting_minutes": 0.2, "article": 0.7, "essay": 0.6,
    "ppt_outline": 0.4,
}

# Max tokens per length
MAX_TOKENS_MAP = {"short": 2000, "medium": 4000, "long": 8000}

# ---------------------------------------------------------------------------
# Layer 1: Global System Prompt
# ---------------------------------------------------------------------------

L1_GLOBAL = """你是由普罗米修斯系统创建的专业文档写作助手。

你的能力：
- 根据用户提供的主题和要点，生成结构清晰、内容准确的各类文档
- 支持 7 种文档类型：周报、邮件、技术文档、会议纪要、文章、短文、PPT 大纲
- 适应用户的个人写作偏好和风格

你的行为约束：
- 只根据用户提供的要点生成内容，不编造未提供的事实或数据
- 如果要点不足以生成完整文档，基于要点合理扩展但保持事实准确
- 始终使用中文输出（除非用户特别指定其他语言）
- 输出格式为 Markdown
- 不输出与文档无关的寒暄、解释或元评论
"""

# ---------------------------------------------------------------------------
# Template loading (Layer 2)
# ---------------------------------------------------------------------------

def _load_template(doc_type: str) -> str:
    """Load a document-type template.

    查找顺序：用户自定义目录 -> 内置目录
    用户在 ~/.function-router/prometheus/templates/ 放置同名 .md 文件即可覆盖内置模板。
    """
    # 优先查找用户自定义模板
    user_path = USER_TEMPLATES_DIR / f"{doc_type}.md"
    if user_path.exists():
        return user_path.read_text(encoding="utf-8")

    # 回退到内置模板
    path = TEMPLATES_DIR / f"{doc_type}.md"
    if not path.exists():
        logger.warning("Template not found: %s, using minimal template", path)
        return f"## 文档类型：{doc_type}\n\n请根据用户提供的主题和要点生成结构清晰的文档。"
    return path.read_text(encoding="utf-8")


def _list_templates() -> list[str]:
    """List available template names (内置 + 用户自定义)."""
    names: set[str] = set()
    if TEMPLATES_DIR.exists():
        names.update(f.stem for f in TEMPLATES_DIR.glob("*.md"))
    if USER_TEMPLATES_DIR.exists():
        names.update(f.stem for f in USER_TEMPLATES_DIR.glob("*.md"))
    return sorted(names)


# ---------------------------------------------------------------------------
# Preference retrieval (Layer 3a)
# ---------------------------------------------------------------------------

def _recall_preferences(context: str) -> dict[str, str]:
    """Recall user writing preferences from memory engine.

    Returns a dict of preference key -> value, or empty dict on failure.
    """
    if not Path(MEMORY_ENGINE).exists():
        logger.debug("Memory engine not found at %s, skipping preference recall", MEMORY_ENGINE)
        return {}

    try:
        result = subprocess.run(
            ["python3", MEMORY_ENGINE, "recall",
             "--query", f"writing {context}",
             "--limit", "5"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.debug("Memory recall failed: %s", result.stderr[:200])
            return {}

        data = json.loads(result.stdout)
        prefs: dict[str, str] = {}

        for mem in data.get("memories", []):
            content = mem.get("content", "")
            # Try to extract key=value patterns from memory content
            # Support "writing_format: markdown", "preferred_language: zh-CN" etc.
            for match in re.finditer(r'(\w+)\s*[:：]\s*(.+?)(?:\n|$)', content):
                key, val = match.group(1).strip(), match.group(2).strip()
                if key in ("writing_format", "preferred_language", "tone",
                           "structure", "style", "language", "format"):
                    prefs[key] = val

        # Also check preferences table
        result2 = subprocess.run(
            ["python3", MEMORY_ENGINE, "list_prefs"],
            capture_output=True, text=True, timeout=10,
        )
        if result2.returncode == 0:
            pref_data = json.loads(result2.stdout)
            for p in pref_data.get("preferences", []):
                key = p.get("key", "")
                if key in ("writing_format", "preferred_language", "tone",
                           "structure", "style", "language", "format"):
                    prefs[key] = p.get("value", "")

        return prefs

    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("Preference recall error: %s", e)
        return {}


def _format_preferences(prefs: dict[str, str]) -> str:
    """Format preferences dict as XML tags for prompt injection."""
    if not prefs:
        return "<user_preferences>（无特定偏好，使用默认设置）</user_preferences>"

    field_map = {
        "preferred_language": "language",
        "language": "language",
        "writing_format": "format",
        "format": "format",
        "tone": "tone",
        "structure": "structure",
        "style": "style",
    }

    lines = ["<user_preferences>"]
    for key, label in field_map.items():
        val = prefs.get(key)
        if val:
            lines.append(f"  <{label}>{val}</{label}>")
    lines.append("</user_preferences>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_generate_prompt(
    doc_type: str,
    topic: str,
    key_points: list[str],
    style: str,
    length: str,
    preferences: dict[str, str],
) -> tuple[str, str, dict[str, Any]]:
    """Build system prompt, user prompt, and API kwargs for generate.

    Returns:
        (system_prompt, user_prompt, api_kwargs)
    """
    template = _load_template(doc_type)
    prefs_xml = _format_preferences(preferences)

    system_prompt = f"""{L1_GLOBAL}

---

## 文档模板

{template}

---

## 用户偏好

{prefs_xml}

如果用户偏好与文档模板的结构建议冲突，以用户偏好为准。
"""

    key_points_str = (
        "\n".join(f"- {p}" for p in key_points)
        if key_points
        else "（用户未提供具体要点，请根据主题合理生成）"
    )

    user_prompt = f"""<task>
主题：{topic}
要点：
{key_points_str}
风格：{style}
篇幅：{length}
</task>

请根据以上信息生成{doc_type}文档。先在思考中列出大纲，再生成完整文档。
将文档内容放在 <document></document> 标签内。
"""

    temperature = TEMP_MAP.get(doc_type, 0.4)
    max_tokens = MAX_TOKENS_MAP.get(length, 4000)

    api_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["</document>"],
    }

    return system_prompt, user_prompt, api_kwargs


def _build_polish_prompt(
    text: str,
    goal: str,
    target_language: str,
    preferences: dict[str, str],
) -> tuple[str, str, dict[str, Any]]:
    """Build prompt for polish tool."""

    goal_map = {
        "more_professional": "更专业、正式，使用行业术语",
        "more_concise": "更简洁，去除冗余，精炼表达",
        "more_friendly": "更友好、亲切，语气自然",
        "fix_grammar": "修正语法错误和标点问题",
        "more_technical": "更技术化，增加技术细节和准确性",
    }

    goal_desc = goal_map.get(goal, "通用润色")

    prefs_xml = _format_preferences(preferences)
    lang_instruction = f"目标语言：{target_language}\n" if target_language else ""

    system_prompt = f"""{L1_GLOBAL}

当前任务：文本润色

润色目标：{goal_desc}

{lang_instruction}
## 用户偏好

{prefs_xml}

请直接输出润色后的文本，放在 <document></document> 标签内。
同时，在 <changes></changes> 标签内简要列出主要修改点（中文，分号分隔）。
"""

    user_prompt = f"""请润色以下文本：

<original>
{text}
</original>
"""

    api_kwargs = {
        "temperature": 0.3,
        "max_tokens": MAX_TOKENS_MAP["long"],
        "stop": ["</changes>"],
    }

    return system_prompt, user_prompt, api_kwargs


def _build_translate_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    keep_formatting: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Build prompt for translate tool."""

    source_desc = source_lang if source_lang and source_lang != "auto" else "自动检测"
    format_instr = "保留原文的 Markdown 格式、代码块、列表结构。" if keep_formatting else "无需保留格式，以自然段落输出。"

    system_prompt = f"""你是由普罗米修斯系统创建的专业翻译助手。

你的任务：
- 将文本从 {source_desc} 翻译为 {target_lang}
- {format_instr}
- 翻译要准确、流畅、自然，符合目标语言的表达习惯
- 不添加原文没有的内容，不遗漏原文信息
- 不输出翻译无关的说明或寒暄

将翻译结果放在 <document></document> 标签内。
"""

    user_prompt = f"""请翻译以下文本：

<original>
{text}
</original>
"""

    api_kwargs = {
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS_MAP["long"],
        "stop": ["</document>"],
    }

    return system_prompt, user_prompt, api_kwargs


# ---------------------------------------------------------------------------
# Humanize - AI pattern dictionaries & rule engine
# ---------------------------------------------------------------------------

EN_AI_VOCABULARY = {
    "delve into": "examine", "delve": "examine",
    "crucial": "important", "pivotal": "key",
    "vibrant": "active", "tapestry": "mix",
    "underscore": "show", "showcase": "display",
    "leverage": "use", "foster": "encourage",
    "enhance": "improve", "intricate": "complex",
    "testament": "proof", "garner": "get",
    "seamless": "smooth", "compelling": "strong",
    "landscape": "field", "interplay": "interaction",
    "realm": "area", "enduring": "lasting",
    "profound": "deep", "boasts": "has",
    "offers": "has", "serves as": "is",
    "stands as": "is", "represents": "is",
    "features": "has",
}

ZH_AI_VOCABULARY = {
    "至关重要": "很重要", "赋能": "支持", "助力": "帮助",
    "深入探讨": "讨论", "在当今时代": "现在",
    "综上所述": "总之", "值得注意的是": "",
    "众所周知": "", "不可或缺": "必需",
    "生态": "体系", "矩阵": "系统", "闭环": "循环",
    "抓手": "手段",
}

PROMOTIONAL_WORDS_EN = {
    "breathtaking": "impressive", "stunning": "striking",
    "nestled": "located", "must-visit": "worth visiting",
    "renowned": "known", "rich heritage": "long history",
    "natural beauty": "scenery",
}

SIGNPOSTING_EN = [
    "Let's dive in", "Let's explore", "Let's break this down",
    "Here's what you need to know", "Without further ado",
    "Now let's look at",
]

SIGNPOSTING_ZH = [
    "让我们深入探讨", "让我们来看看", "接下来我们将",
]

PUNCTUATION_RULES = {
    "\u2014": ",",    # em dash -> comma
    "\u2013": "-",    # en dash -> hyphen
    "\u201c": "\"",   # curly left quote -> straight
    "\u201d": "\"",   # curly right quote -> straight
    "\u2018": "'",    # curly left single
    "\u2019": "'",    # curly right single
}

# Regex patterns for Chinese cliché removal
ZH_CLICHE_PATTERNS = [
    (re.compile(r'随着[^，。、]{1,20}的发展[，,]?'), ''),
    (re.compile(r'在当今[^，。、]{0,10}背景下[，,]?'), '现在'),
    (re.compile(r'首先[，,][^。]{1,50}其次[，,][^。]{1,50}最后[，,]'), ''),
]

# Regex patterns for English cliché removal
EN_CLICHE_PATTERNS = [
    (re.compile(r'marks? a (?:pivotal|significant|important) moment', re.IGNORECASE), ''),
    (re.compile(r'stands? as a testament to', re.IGNORECASE), 'shows'),
    (re.compile(r'underscores? the significance of', re.IGNORECASE), 'shows the importance of'),
    (re.compile(r'in today\'?s (?:world|society|landscape)', re.IGNORECASE), 'today'),
    (re.compile(r'the future looks bright', re.IGNORECASE), ''),
    (re.compile(r'exciting times lie ahead', re.IGNORECASE), ''),
]


class HumanizeRuleEngine:
    """Rule-based AI pattern removal (Light level humanization)."""

    def __init__(self):
        self.en_vocab = EN_AI_VOCABULARY
        self.zh_vocab = ZH_AI_VOCABULARY
        self.promotional = PROMOTIONAL_WORDS_EN
        self.signposting_en = SIGNPOSTING_EN
        self.signposting_zh = SIGNPOSTING_ZH
        self.punct_rules = PUNCTUATION_RULES
        self.zh_cliche_patterns = ZH_CLICHE_PATTERNS
        self.en_cliche_patterns = EN_CLICHE_PATTERNS

    def detect_language(self, text: str) -> str:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total = len(text)
        if total == 0:
            return "en"
        return "zh" if chinese_chars > total * 0.3 else "en"

    def fix_punctuation(self, text: str) -> tuple[str, int]:
        count = 0
        for old, new in self.punct_rules.items():
            n = text.count(old)
            if n:
                text = text.replace(old, new)
                count += n
        return text, count

    def replace_ai_vocabulary(self, text: str, lang: str) -> tuple[str, int]:
        vocab = self.zh_vocab if lang == "zh" else self.en_vocab
        count = 0
        for ai_word, replacement in vocab.items():
            if ai_word in text:
                n = text.count(ai_word)
                text = text.replace(ai_word, replacement)
                count += n
        text = re.sub(r'  +', ' ', text)
        return text, count

    def remove_signposting(self, text: str, lang: str) -> tuple[str, int]:
        phrases = self.signposting_zh if lang == "zh" else self.signposting_en
        count = 0
        for phrase in phrases:
            if phrase in text:
                n = text.count(phrase)
                text = text.replace(phrase, "")
                count += n
        text = re.sub(r'^\s*[,，。]\s*', '', text, flags=re.MULTILINE)
        return text, count

    def downgrade_promotional(self, text: str, lang: str) -> tuple[str, int]:
        count = 0
        if lang == "en":
            for word, replacement in self.promotional.items():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    count += len(matches)
                    text = pattern.sub(replacement, text)
        return text, count

    def remove_cliches(self, text: str, lang: str) -> tuple[str, int]:
        patterns = self.zh_cliche_patterns if lang == "zh" else self.en_cliche_patterns
        count = 0
        for pattern, replacement in patterns:
            new_text = pattern.sub(replacement, text)
            if new_text != text:
                count += 1
                text = new_text
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text, count

    def check_em_dash_zero(self, text: str) -> bool:
        return "\u2014" not in text and "\u2013" not in text

    def calculate_burstiness(self, text: str) -> float:
        sentences = re.split(r'[。.!?！？\n]', text)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(lengths) < 2:
            return 0.0
        mean = sum(lengths) / len(lengths)
        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        return variance ** 0.5

    def humanize_light(self, text: str) -> dict[str, Any]:
        lang = self.detect_language(text)
        changes = []

        text, n = self.fix_punctuation(text)
        if n:
            changes.append(f"修正标点 {n} 处")

        text, n = self.replace_ai_vocabulary(text, lang)
        if n:
            changes.append(f"替换AI高频词 {n} 处")

        text, n = self.remove_signposting(text, lang)
        if n:
            changes.append(f"删除信号标注 {n} 处")

        text, n = self.downgrade_promotional(text, lang)
        if n:
            changes.append(f"降级推广语言 {n} 处")

        text, n = self.remove_cliches(text, lang)
        if n:
            changes.append(f"删除套话 {n} 处")

        return {
            "humanized": text.strip(),
            "changes_summary": "; ".join(changes) if changes else "无明显AI特征需修改",
            "stats": {
                "em_dash_zero": self.check_em_dash_zero(text),
                "burstiness": round(self.calculate_burstiness(text), 4),
            },
        }


# ---------------------------------------------------------------------------
# Humanize prompt builders
# ---------------------------------------------------------------------------

MEDIUM_SYSTEM_PROMPT = """你是一个专业文本编辑。请改写文本使其更自然。

严格要求：
1. 保留所有核心信息和事实
2. 用简单动词替代华丽动词（is/are/has 而非 serves as/boasts）
3. 打破"三元组"结构（A, B, and C -> 自由表达）
4. 将 -ing 短语堆砌改为独立句子
5. 被动语态改为主动语态
6. 句子长度交替变化（短句 <10词，长句 >20词）
7. 禁用词汇：delve, crucial, pivotal, vibrant, tapestry, underscore, showcase, leverage, foster, enhance, intricate, testament, landscape, interplay, seamless, compelling
8. 禁用 em dash（—）和 en dash（–）
9. 不添加原文没有的信息
10. 保持段落数和大致结构

将改写结果放在 <document></document> 标签内。
"""

HEAVY_SYSTEM_PROMPT = """你是一个专业写手。请完全改写文本使其像真人写的。

核心要求：
1. 保留核心信息，但完全改变表达方式
2. 像在跟同事说话一样自然
3. 可以有个人观点和不确定性
4. 用具体细节替代抽象概括
5. 句子长短自由变化
6. 允许轻微"不完美"——口语化、半成型的想法
7. 绝对禁止：
   - em dash（—）和 en dash（–）
   - AI高频词：delve, crucial, tapestry, vibrant, underscore, showcase, foster, enhance, landscape, testament, intricate, seamless, compelling
   - Rule of Three（三元组）
   - "serves as" 替代 "is"
   - 推广语言：breathtaking, nestled, stunning, vibrant, renowned
   - 模糊归因：experts argue, industry reports
   - 通用结尾：future looks bright, exciting times
   - 信号标注：Let's dive in, here's what you need
   - -ing 短语堆砌
   - 否定平行：not only...but also, not just...it's
   - 格言公式：X is the Y of Z

将改写结果放在 <document></document> 标签内。
"""

AUDIT_PROMPT = """请审查以下改写文本，列出所有残留的AI写作痕迹。

检查清单：
1. 是否有 em dash（—）或 en dash（–）？
2. 是否有 AI 高频词？
3. 是否有 Rule of Three？
4. 是否有推广性语言？
5. 是否有模糊归因？
6. 句子长度是否均匀？
7. 是否有通用积极结尾？
8. 是否有信号标注？
9. 是否有 -ing 短语堆砌？
10. 是否有否定平行结构？

只列出发现的问题，如果没有问题请回答"无残留AI痕迹"。

改写文本：
{text}
"""


# ---------------------------------------------------------------------------
# Upstream LLM call
# ---------------------------------------------------------------------------

def _call_upstream_llm(
    system_prompt: str,
    user_prompt: str,
    api_kwargs: dict[str, Any],
    upstream_url: str | None = None,
    upstream_model: str | None = None,
    upstream_key: str | None = None,
) -> str:
    """Call upstream LLM via OpenAI-compatible API.

    Uses httpx if available, falls back to urllib.
    Returns the assistant message content text.
    Raises RuntimeError on failure.
    """
    url = upstream_url or UPSTREAM_URL
    model = upstream_model or UPSTREAM_MODEL
    key = upstream_key or UPSTREAM_KEY

    if not url or not model:
        raise RuntimeError(
            "upstream LLM not configured. Set WRITING_UPSTREAM_URL and "
            "WRITING_UPSTREAM_MODEL environment variables."
        )

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **api_kwargs,
    }

    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    if HAS_HTTPX:
        resp = httpx.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    else:
        # Fallback: urllib
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"upstream HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"upstream returned no choices: {data}")

    return choices[0]["message"]["content"]


# ---------------------------------------------------------------------------
# Output extraction helpers
# ---------------------------------------------------------------------------

def _extract_tag(text: str, tag: str) -> str:
    """Extract content between XML tags. Falls back to full text if not found."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_document(text: str) -> str:
    """Extract content from <document> tags, stripping prefill if present."""
    return _extract_tag(text, "document")


def _extract_changes(text: str) -> str:
    """Extract content from <changes> tags."""
    return _extract_tag(text, "changes")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def generate(
    doc_type: str,
    topic: str,
    key_points: list[str] | None = None,
    style: str = "formal",
    length: str = "medium",
    upstream_url: str | None = None,
    upstream_model: str | None = None,
    upstream_key: str | None = None,
) -> dict[str, Any]:
    """Generate a document from topic + key_points."""

    if doc_type not in VALID_DOC_TYPES:
        return {"error": f"invalid doc_type: {doc_type}. Valid: {sorted(VALID_DOC_TYPES)}"}
    if not topic or not topic.strip():
        return {"error": "topic must not be empty"}

    style = style if style in VALID_STYLES else "formal"
    length = length if length in VALID_LENGTHS else "medium"
    key_points = key_points or []

    # Recall user preferences
    prefs = _recall_preferences(doc_type)

    # Build prompt
    system_prompt, user_prompt, api_kwargs = _build_generate_prompt(
        doc_type, topic, key_points, style, length, prefs,
    )

    # Call upstream LLM
    try:
        raw_output = _call_upstream_llm(
            system_prompt, user_prompt, api_kwargs,
            upstream_url, upstream_model, upstream_key,
        )
    except Exception as e:
        return {"error": f"upstream LLM call failed: {e}", "doc_type": doc_type, "topic": topic}

    document = _extract_document(raw_output)

    return {
        "result": "ok",
        "doc_type": doc_type,
        "topic": topic,
        "document": document,
        "format": "markdown",
        "preferences_applied": bool(prefs),
    }


def polish(
    text: str,
    goal: str = "more_professional",
    target_language: str = "",
    upstream_url: str | None = None,
    upstream_model: str | None = None,
    upstream_key: str | None = None,
) -> dict[str, Any]:
    """Polish/refine existing text."""

    if not text or not text.strip():
        return {"error": "text must not be empty"}
    goal = goal if goal in VALID_GOALS else "more_professional"

    prefs = _recall_preferences("polish")
    system_prompt, user_prompt, api_kwargs = _build_polish_prompt(
        text, goal, target_language, prefs,
    )

    try:
        raw_output = _call_upstream_llm(
            system_prompt, user_prompt, api_kwargs,
            upstream_url, upstream_model, upstream_key,
        )
    except Exception as e:
        return {"error": f"upstream LLM call failed: {e}"}

    polished = _extract_document(raw_output)
    changes = _extract_changes(raw_output)

    return {
        "result": "ok",
        "polished": polished,
        "changes_summary": changes if changes != raw_output.strip() else "",
        "original_length": len(text),
        "polished_length": len(polished),
    }


def translate(
    text: str,
    target_lang: str,
    source_lang: str = "auto",
    keep_formatting: bool = True,
    upstream_url: str | None = None,
    upstream_model: str | None = None,
    upstream_key: str | None = None,
) -> dict[str, Any]:
    """Translate text between languages."""

    if not text or not text.strip():
        return {"error": "text must not be empty"}
    if not target_lang or not target_lang.strip():
        return {"error": "target_lang must not be empty"}

    system_prompt, user_prompt, api_kwargs = _build_translate_prompt(
        text, source_lang, target_lang, keep_formatting,
    )

    try:
        raw_output = _call_upstream_llm(
            system_prompt, user_prompt, api_kwargs,
            upstream_url, upstream_model, upstream_key,
        )
    except Exception as e:
        return {"error": f"upstream LLM call failed: {e}"}

    translated = _extract_document(raw_output)

    return {
        "result": "ok",
        "translated": translated,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "original_length": len(text),
        "translated_length": len(translated),
    }


def humanize(
    text: str,
    intensity: str = "medium",
    preserve_formatting: bool = True,
    upstream_url: str | None = None,
    upstream_model: str | None = None,
    upstream_key: str | None = None,
) -> dict[str, Any]:
    """Remove AI writing patterns with three intensity levels.

    Light:  Rule-based only (punctuation, vocab, signposting, cliches)
    Medium: Rule pre-pass + LLM single-round rewrite
    Heavy:  Rule pre-pass + LLM Draft + Audit + Revise
    """

    if not text or not text.strip():
        return {"error": "text must not be empty"}
    intensity = intensity if intensity in VALID_INTENSITIES else "medium"

    engine = HumanizeRuleEngine()
    original_burstiness = engine.calculate_burstiness(text)
    changes: list[str] = []

    # Step 1: Rule pre-pass (all levels)
    result = engine.humanize_light(text)
    preprocessed = result["humanized"]
    if "无明显AI特征需修改" not in result["changes_summary"]:
        changes.append(f"规则预清洗: {result['changes_summary']}")

    if intensity == "light":
        final_text = preprocessed
        stats = {
            "em_dash_zero": engine.check_em_dash_zero(final_text),
            "burstiness": round(engine.calculate_burstiness(final_text), 4),
            "original_burstiness": round(original_burstiness, 4),
            "burstiness_improved": engine.calculate_burstiness(final_text) > original_burstiness,
        }
        return {
            "result": "ok",
            "humanized": final_text,
            "changes_summary": " | ".join(changes) if changes else "无明显AI特征需修改",
            "intensity": intensity,
            "stats": stats,
        }

    # Step 2: LLM rewrite (medium / heavy)
    if intensity == "medium":
        try:
            raw_output = _call_upstream_llm(
                MEDIUM_SYSTEM_PROMPT, preprocessed,
                {"temperature": 0.4, "max_tokens": MAX_TOKENS_MAP["long"], "stop": ["</document>"]},
                upstream_url, upstream_model, upstream_key,
            )
            rewritten = _extract_document(raw_output)
            changes.append("LLM句式重写完成")
        except Exception as e:
            return {"error": f"upstream LLM call failed (medium): {e}", "humanized": preprocessed,
                    "changes_summary": " | ".join(changes)}

    elif intensity == "heavy":
        rewritten = preprocessed  # fallback
        try:
            # Draft
            raw_output = _call_upstream_llm(
                HEAVY_SYSTEM_PROMPT, preprocessed,
                {"temperature": 0.7, "max_tokens": MAX_TOKENS_MAP["long"], "stop": ["</document>"]},
                upstream_url, upstream_model, upstream_key,
            )
            draft = _extract_document(raw_output)
            changes.append("LLM第一轮改写完成")

            # Audit
            audit_input = AUDIT_PROMPT.format(text=draft)
            audit_result = _call_upstream_llm(
                "你是文本审查专家。", audit_input,
                {"temperature": 0.2, "max_tokens": 2000},
                upstream_url, upstream_model, upstream_key,
            )

            # Revise if issues found
            if "无残留AI痕迹" not in audit_result:
                revise_prompt = f"""根据以下审查意见修订文本：

审查意见：
{audit_result}

待修订文本：
{draft}

请修订所有指出的问题，输出终稿。将结果放在 <document></document> 标签内。"""
                raw_output = _call_upstream_llm(
                    "你是文本修订专家。", revise_prompt,
                    {"temperature": 0.3, "max_tokens": MAX_TOKENS_MAP["long"], "stop": ["</document>"]},
                    upstream_url, upstream_model, upstream_key,
                )
                rewritten = _extract_document(raw_output)
                issue_count = audit_result.count("\n")
                changes.append(f"LLM自审修订完成（发现 {issue_count} 个问题）")
            else:
                rewritten = draft
                changes.append("LLM自审通过，无残留痕迹")

        except Exception as e:
            return {"error": f"upstream LLM call failed (heavy): {e}", "humanized": preprocessed,
                    "changes_summary": " | ".join(changes)}

    # Step 3: Rule post-pass
    post_result = engine.humanize_light(rewritten)
    final_text = post_result["humanized"]
    if "无明显AI特征需修改" not in post_result["changes_summary"]:
        changes.append(f"规则后清洗: {post_result['changes_summary']}")

    # Step 4: Quality stats
    final_burstiness = engine.calculate_burstiness(final_text)
    stats = {
        "em_dash_zero": engine.check_em_dash_zero(final_text),
        "burstiness": round(final_burstiness, 4),
        "original_burstiness": round(original_burstiness, 4),
        "burstiness_improved": final_burstiness > original_burstiness,
    }

    return {
        "result": "ok",
        "humanized": final_text,
        "changes_summary": " | ".join(changes),
        "intensity": intensity,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_json(obj: dict[str, Any]):
    print(json.dumps(obj, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Writing engine for MTClaw - generate, polish, translate & humanize documents."
    )
    parser.add_argument(
        "tool",
        choices=["writing_generate", "writing_polish", "writing_translate", "writing_humanize"],
        help="Tool to execute",
    )
    parser.add_argument(
        "--upstream-url", default=None,
        help="Override upstream LLM API URL",
    )
    parser.add_argument(
        "--upstream-model", default=None,
        help="Override upstream LLM model name",
    )
    parser.add_argument(
        "--upstream-key", default=None,
        help="Override upstream LLM API key",
    )
    args = parser.parse_args()

    # Read JSON input from stdin
    try:
        raw_input = sys.stdin.read()
        params = json.loads(raw_input) if raw_input.strip() else {}
    except json.JSONDecodeError as e:
        _print_json({"error": f"invalid JSON input: {e}"})
        sys.exit(1)

    try:
        if args.tool == "writing_generate":
            result = generate(
                doc_type=params["doc_type"],
                topic=params["topic"],
                key_points=params.get("key_points", []),
                style=params.get("style", "formal"),
                length=params.get("length", "medium"),
                upstream_url=args.upstream_url,
                upstream_model=args.upstream_model,
                upstream_key=args.upstream_key,
            )
        elif args.tool == "writing_polish":
            result = polish(
                text=params["text"],
                goal=params.get("goal", "more_professional"),
                target_language=params.get("target_language", ""),
                upstream_url=args.upstream_url,
                upstream_model=args.upstream_model,
                upstream_key=args.upstream_key,
            )
        elif args.tool == "writing_translate":
            result = translate(
                text=params["text"],
                target_lang=params["target_lang"],
                source_lang=params.get("source_lang", "auto"),
                keep_formatting=params.get("keep_formatting", True),
                upstream_url=args.upstream_url,
                upstream_model=args.upstream_model,
                upstream_key=args.upstream_key,
            )
        elif args.tool == "writing_humanize":
            result = humanize(
                text=params["text"],
                intensity=params.get("intensity", "medium"),
                preserve_formatting=params.get("preserve_formatting", True),
                upstream_url=args.upstream_url,
                upstream_model=args.upstream_model,
                upstream_key=args.upstream_key,
            )
        else:
            result = {"error": f"unknown tool: {args.tool}"}

        _print_json(result)

    except KeyError as e:
        _print_json({"error": f"missing required parameter: {e}"})
        sys.exit(1)
    except Exception as e:
        logger.exception("Engine error")
        _print_json({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("WRITING_LOG_LEVEL", "WARNING"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
