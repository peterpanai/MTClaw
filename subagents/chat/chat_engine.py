#!/usr/bin/env python3
"""闲聊陪伴 Subagent 引擎。

路由模型直回，不走上游 LLM。利用路由模型轻量高效特性实现秒级回复。
"""

from __future__ import annotations

import json
import os
import sys
import re
import httpx

# ── 配置 ──────────────────────────────────────────────

ROUTING_URL = os.environ.get("ROUTING_URL", "")
ROUTING_MODEL = os.environ.get("ROUTING_MODEL", "")
ROUTING_KEY = os.environ.get("ROUTING_KEY", "")

# ── 闲聊意图识别 ─────────────────────────────────────

# 问候类关键词
GREETING_KEYWORDS = {"你好", "嗨", "hey", "hi", "早上好", "晚安", "晚安",
                     "中午好", "下午好", "哈喽", "在吗"}

# 情感类关键词
EMOTION_KEYWORDS = {"开心", "难过", "无聊", "好累", "烦死了", "郁闷",
                    "生气", "感动", "兴奋", "焦虑", "压力", "崩溃"}

# 娱乐类关键词
ENTERTAINMENT_KEYWORDS = {"笑话", "故事", "谜语", "冷笑话", "有趣",
                          "好玩", "段子", "脑筋急转弯"}

# 简单问答关键词
SIMPLE_QA_KEYWORDS = {"今天星期几", "你叫什么", "天气怎么样", "几点了",
                      "今天是几号", "你叫什么名字"}

# complex 消息禁止词（出现这些词时绝对不走闲聊）
COMPLEX_INDICATORS = {
    # 技术动词
    "代码", "函数", "变量", "编译", "部署", "调试", "架构", "接口",
    "数据库", "服务器", "配置", "环境", "依赖", "版本",
    # 分析类
    "分析", "统计", "对比", "区别", "优缺点", "为什么", "怎么做",
    "原理", "机制", "流程",
    # 文件相关
    ".py", ".js", ".md", ".csv", ".json", ".yaml", ".txt", ".pdf",
    "/path/", "/home/", "/var/", "/etc/",
}

# 文件扩展名模式
FILE_EXT_PATTERN = re.compile(r'\.\w{1,5}$', re.IGNORECASE)


def is_chat_intent(message: str) -> bool:
    """判断消息是否为纯闲聊意图。

    5 条规则全部满足才返回 True：
    1. 无文件路径引用
    2. 无数据查询意图
    3. 无知识检索需求
    4. 消息长度 < 200 字
    5. 包含社交/情感/寒暄语义
    """
    if not message or not message.strip():
        return False

    # 规则 4: 消息长度
    if len(message) > 200:
        return False

    # 规则 1: 无文件路径引用
    if FILE_EXT_PATTERN.search(message):
        return False
    if "/" in message and any(ext in message for ext in [".py", ".js", ".md", ".csv"]):
        return False

    # 误判保护：complex 消息绝对禁止
    msg_lower = message.lower()
    for indicator in COMPLEX_INDICATORS:
        if indicator in msg_lower or indicator in message:
            return False

    # 规则 5: 包含社交/情感/寒暄语义
    all_keywords = (GREETING_KEYWORDS | EMOTION_KEYWORDS |
                    ENTERTAINMENT_KEYWORDS | SIMPLE_QA_KEYWORDS)
    has_social = any(kw in message for kw in all_keywords)

    # 如果没有明确社交关键词，但消息很短且无害，也允许
    if not has_social:
        # 检查是否是纯寒暄（"在吗""好的""谢谢"等）
        harmless = {"在吗", "好的", "谢谢", " thanks", "ok", "嗯", "哦",
                    "哈哈", "嘿嘿", "呵呵", "好的呀", "好呀", "可以"}
        if any(kw in message for kw in harmless):
            has_social = True

    return has_social


def detect_mood(message: str) -> str:
    """自动检测用户情绪，返回 mood 策略。"""
    if any(kw in message for kw in EMOTION_KEYWORDS):
        # 负面情绪
        negative = {"难过", "无聊", "好累", "烦死了", "郁闷", "生气",
                    "焦虑", "压力", "崩溃"}
        if any(kw in message for kw in negative):
            return "comfort"
        # 正面情绪
        positive = {"开心", "感动", "兴奋"}
        if any(kw in message for kw in positive):
            return "casual"
        return "comfort"

    if any(kw in message for kw in ENTERTAINMENT_KEYWORDS):
        return "humor"

    # 好奇类
    curious = {"为什么", "怎么", "什么", "哪里", "哪个", "有趣"}
    if any(kw in message for kw in curious):
        return "curious"

    return "casual"


def get_memory_context() -> str:
    """从 memory_engine 获取用户画像上下文。"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
        from memory_engine import recall
        result = recall({"context": "user profile and preferences", "top_k": 5})
        if result and isinstance(result, dict) and result.get("memories"):
            lines = []
            for mem in result["memories"]:
                lines.append(f"- {mem.get('key', '')}: {mem.get('value', '')}")
            return "\n".join(lines)
    except Exception:
        pass
    return ""


def build_prompt(mood: str, memory_context: str) -> str:
    """构造路由模型直回的 system prompt。"""
    mood_strategies = {
        "casual": "自然日常对话，语气轻松友好。像一个老朋友在聊天。",
        "humor": "优先讲笑话、段子、趣事，风趣幽默。让用户开心起来。",
        "comfort": "共情倾听，温暖安慰。先理解用户的感受，再给予支持。",
        "curious": "拓展话题，提出有趣的问题。激发用户的好奇心。",
    }

    strategy = mood_strategies.get(mood, mood_strategies["casual"])

    prompt = f"""你是普罗米修斯，一个友好、温暖的 AI 助手。

对话风格：{strategy}

要求：
1. 回复简洁自然，不要太长（通常 1-3 句话）
2. 不要使用 emoji 表情符号过多
3. 不要像 AI 一样机械回复
4. 用中文回复"""

    if memory_context:
        prompt += f"\n\n用户画像：\n{memory_context}\n\n请自然地融入这些信息到对话中。"

    return prompt


def call_routing_model(system_prompt: str, user_message: str) -> str:
    """调用路由模型生成回复。"""
    if not ROUTING_URL or not ROUTING_MODEL:
        return "你好呀！我是普罗米修斯，有什么可以帮你的吗？"

    payload = {
        "model": ROUTING_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,  # 闲聊需要一定随机性
        "max_tokens": 512,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {ROUTING_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"{ROUTING_URL.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return "嗯嗯，我在听呢~"
    except Exception as e:
        return f"抱歉，我刚才走神了。你说的'{user_message[:20]}'，能再说一遍吗？"


def chat_light(params: dict) -> dict:
    """闲聊直回主函数。"""
    mood = params.get("mood", "auto")
    memory_inject = params.get("memory_inject", True)

    # 从参数中获取用户消息（由 FR 注入）
    user_message = params.get("_user_message", "")

    # mood 自动检测
    if mood == "auto":
        mood = detect_mood(user_message)

    # 获取记忆上下文
    memory_context = ""
    if memory_inject:
        memory_context = get_memory_context()

    # 构造 prompt
    system_prompt = build_prompt(mood, memory_context)

    # 调用路由模型
    reply = call_routing_model(system_prompt, user_message)

    return {
        "status": "ok",
        "mood": mood,
        "reply": reply,
        "memory_injected": bool(memory_context),
    }


# ── CLI 入口 ──────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "缺少命令参数"}))
        sys.exit(1)

    command = sys.argv[1]

    try:
        params = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        params = {}

    if command == "chat":
        result = chat_light(params)
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps({"error": f"未知命令: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
