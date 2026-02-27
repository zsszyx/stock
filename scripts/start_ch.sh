#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "🚀 启动 ClickHouse (全量同步模式)..."
sed "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" configs/clickhouse/clickhouse_config.xml.template > configs/clickhouse/clickhouse_config.xml

pkill -9 clickhouse 2>/dev/null
rm -f ch_data/status 2>/dev/null
mkdir -p logs ch_data/tmp ch_data/data ch_data/metadata ch_data/store

# 使用绝对路径启动
clickhouse server --config-file "$PROJECT_ROOT/configs/clickhouse/clickhouse_config.xml" --daemon

sleep 5
if curl -s http://localhost:8123 > /dev/null; then
    echo "✅ ClickHouse 已成功启动 (8123)"
else
    echo "❌ 启动失败，尝试前台诊断..."
    clickhouse server --config-file "$PROJECT_ROOT/configs/clickhouse/clickhouse_config.xml" 2>&1 | head -n 20
    exit 1
fi
