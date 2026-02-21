#!/bin/bash

# 获取项目根目录 (脚本所在目录的上一级)
PROJECT_ROOT="/Users/zsy/stock/stock"
cd "$PROJECT_ROOT" || exit 1

echo "------------------------------------------------"
echo "🚀 正在启动 ClickHouse 服务..."
echo "------------------------------------------------"

# 1. 清理旧进程
echo "🧹 检查并清理旧的 ClickHouse 进程..."
pkill -9 clickhouse 2>/dev/null
rm -f ch_data/status 2>/dev/null

# 2. 确保日志目录存在
mkdir -p logs

# 3. 启动服务 (后台运行)
echo "📂 使用配置文件: configs/clickhouse/clickhouse_config.xml"
clickhouse server --config-file configs/clickhouse/clickhouse_config.xml --path ch_data/ --daemon

# 4. 验证端口响应
echo "⏳ 等待服务就绪 (检查 8123 端口)..."
MAX_RETRIES=10
RETRY_COUNT=0
SUCCESS=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8123 > /dev/null; then
        echo "✅ ClickHouse 启动成功! 端口 8123 已就绪。"
        SUCCESS=true
        break
    fi
    echo "... 仍在尝试连接 ($((RETRY_COUNT + 1))/$MAX_RETRIES) ..."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$SUCCESS" = false ]; then
    echo "❌ 启动超时或失败。请检查日志: logs/clickhouse-server.err.log"
    exit 1
fi

echo "------------------------------------------------"
echo "💡 提示: 运行 'tail -f logs/clickhouse-server.log' 查看实时日志"
echo "------------------------------------------------"
