"""conftest.py - Prometheus 测试公共 fixtures。"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 将 subagents 目录加入 sys.path
SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """临时数据目录，测试后自动清理。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    os.environ["PROMETHEUS_DATA_DIR"] = str(data_dir)
    os.environ["RAG_DATA_DIR"] = str(data_dir)
    yield data_dir
    os.environ.pop("PROMETHEUS_DATA_DIR", None)
    os.environ.pop("RAG_DATA_DIR", None)
