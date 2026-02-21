# Copilot Instructions for AI Agents

## 项目概览
- 本项目为基于Python的股票量化交易系统，核心功能包括多源数据采集、数据清洗、数据库管理、量化策略实现与结果输出。
- 主要目录：
  - `init/`：数据采集、准备、清洗、数据库操作、代理配置
  - `strategy/`：量化交易策略实现（如“二波”策略）
  - `quantization/`：因子分析与量化工具
  - `test.py`：主程序入口，负责调度数据准备与策略执行

## 关键开发模式与约定
- 数据流：数据采集→清洗→入库（SQLite）→策略分析→结果输出（Excel）
- 多数据源冗余：如akshare、efinance、baostock，自动切换，详见`init/data_prepare.py`
- 策略开发：每种策略单独放在`strategy/`，如`erbo.py`，需实现策略筛选与结果标记
- 数据库操作集中在`init/sql_op.py`，避免在策略中直接操作数据库
- 结果统一输出到`result.xlsx`，格式由策略模块定义

## 典型工作流
1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python test.py`
3. 查看输出：`result.xlsx`

## 扩展建议
- 新数据源：在`init/`下新建采集脚本，并在`data_prepare.py`注册
- 新策略：在`strategy/`下新建文件，主入口在`test.py`中调用
- 新因子分析：在`quantization/`下实现，供策略或主程序调用

## 重要注意事项
- 数据接口有频率限制，已实现多源备份与代理池（见`proxy_pool.yaml`）
- 首次运行数据量大，耗时较长
- 某些股票数据可能缺失，需容错处理

## 参考文件
- `README_运行指南.md`：详细运行与开发说明
- `init/data_prepare.py`、`strategy/erbo.py`、`test.py`：核心流程范例

---
如需自定义AI代理行为，请遵循上述结构与约定，保持模块解耦与数据流清晰。