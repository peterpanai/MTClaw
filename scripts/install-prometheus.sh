#!/bin/bash
set -euo pipefail

# Prometheus Subagent 安装脚本
# 在 MTClaw 安装完成后运行，安装 5 个 Subagent

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SUBAGENTS_DIR="${REPO_ROOT}/subagents"
TARGET_DIR="${HOME}/.function-router"
CONFIG_PATH="${TARGET_DIR}/config.json"
FUNCTIONS_PATH="${TARGET_DIR}/functions.jsonl"
SCRIPTS_DIR="${TARGET_DIR}/scripts"
PROMETHEUS_DATA="${HOME}/.prometheus"

echo "=== Prometheus Subagent 安装 ==="
echo ""

# 1. 检查 MTClaw 是否已安装
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "✗ MTClaw 未安装，请先运行 MTClaw 的 install.sh"
    exit 1
fi

# 2. 安装 Python 依赖
echo "── 安装 Prometheus Python 依赖 ──"
pip install -r "${SUBAGENTS_DIR}/requirements.txt" 2>&1 | tail -3

# 3. 创建 Prometheus 数据目录
echo "── 创建数据目录 ──"
mkdir -p "${PROMETHEUS_DATA}/data/chroma"
mkdir -p "${PROMETHEUS_DATA}/data/charts"
mkdir -p "${PROMETHEUS_DATA}/logs"
mkdir -p "${PROMETHEUS_DATA}/templates"

# 4. 聚合 functions.jsonl
echo "── 聚合工具定义 ──"
python3 "${SUBAGENTS_DIR}/aggregate_functions.py" --output "${FUNCTIONS_PATH}"
TOOLS_COUNT=$(wc -l < "${FUNCTIONS_PATH}")
echo "  已聚合 ${TOOLS_COUNT} 个工具定义"

# 5. 复制 wrapper 脚本
echo "── 复制 wrapper 脚本 ──"
for subagent in rag memory writing schedule chat; do
    if [ -d "${SUBAGENTS_DIR}/${subagent}/scripts" ]; then
        cp "${SUBAGENTS_DIR}/${subagent}/scripts/"*.sh "${SCRIPTS_DIR}/"
    fi
done
chmod +x "${SCRIPTS_DIR}/"*.sh
SCRIPTS_COUNT=$(ls "${SCRIPTS_DIR}/"*.sh 2>/dev/null | wc -l)
echo "  已复制 ${SCRIPTS_COUNT} 个 wrapper 脚本"

# 6. 复制写作模板
echo "── 复制写作模板 ──"
if [ -d "${SUBAGENTS_DIR}/writing/templates" ]; then
    cp "${SUBAGENTS_DIR}/writing/templates/"*.md "${PROMETHEUS_DATA}/templates/"
    TEMPLATES_COUNT=$(ls "${PROMETHEUS_DATA}/templates/"*.md 2>/dev/null | wc -l)
    echo "  已复制 ${TEMPLATES_COUNT} 个写作模板"
fi

# 7. 初始化数据库
echo "── 初始化数据库 ──"
python3 -c "
import sys
sys.path.insert(0, '${SUBAGENTS_DIR}/memory')
from memory_engine import init_db
init_db('${PROMETHEUS_DATA}/data/prometheus.db')
print('  SQLite 数据库已初始化')
" 2>&1 || echo "  ⚠ 数据库初始化跳过（可能缺少依赖）"

# 8. 设置环境变量提示
echo ""
echo "── 环境变量配置 ──"
echo "请在 ~/.function-router/config.json 中确认以下配置："
echo "  - routing.model: 路由模型名称"
echo "  - routing.api_key: 路由模型 API Key"
echo "  - upstream.model: 上游 LLM 名称"
echo "  - upstream.api_key: 上游 LLM API Key"
echo ""
echo "Prometheus 数据目录: ${PROMETHEUS_DATA}"
echo ""
echo "=== 安装完成 ==="
echo "重启 MTClaw: cd ${REPO_ROOT} && bash scripts/restart_all.sh"
echo "健康检查: curl http://127.0.0.1:18790/health"
echo "工具列表: curl http://127.0.0.1:18790/v1/tools | python3 -m json.tool"
