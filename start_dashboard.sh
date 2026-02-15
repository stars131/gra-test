#!/bin/bash
# 一键启动可视化仪表板

echo "=========================================="
echo "  网络攻击检测可视化系统"
echo "=========================================="

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装"
    exit 1
fi

# 检查Streamlit
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "[提示] 正在安装 Streamlit..."
    pip install streamlit plotly -q
fi

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "[启动] 正在启动仪表板..."
echo "[访问] http://localhost:8501"
echo "[退出] 按 Ctrl+C 停止"
echo "=========================================="

# 启动Streamlit
streamlit run src/visualization/app.py --server.headless true
