# KSP 股票动能策略 - 完整开发与维护文档 (v2.0 稳定版)

这份文档旨在为 KSP (Kinetic Statistical Power) 策略的开发者、维护者和使用者提供全面、深度的技术指导。

---

## 📖 目录

1.  [🚀 系统概述与核心架构](#1-系统概述与核心架构)
    *   [1.1 架构设计图](#11-架构设计图)
    *   [1.2 技术栈](#12-技术栈)
2.  [📊 数据库与数据模块开发 (`stock.database` & `stock.data_context`)](#2-数据库与数据模块开发)
    *   [2.1 ClickHouse 表结构与维护](#21-clickhouse-表结构与维护)
    *   [2.2 数据上下文 (Data Context)](#22-数据上下文)
3.  [🔄 数据流水线任务 (`stock.tasks`)](#3-数据流水线任务)
    *   [3.1 Min5UpdateTask: 原始数据抓取](#31-min5updatetask)
    *   [3.2 DailyAggregationTask: 日线聚合与 POC](#32-dailyaggregationtask)
    *   [3.3 RefreshFactorsTask: 截面排名计算](#33-refreshfactorstask)
4.  [🧠 策略核心逻辑开发 (`stock.strategy` & `stock.selector`)](#4-策略核心逻辑开发)
    *   [4.1 次优区间哲学](#41-次优区间哲学)
    *   [4.2 选股漏斗 (FunnelSelector)](#42-选股漏斗)
    *   [4.3 模块化策略引擎 (ModularKSPCore)](#43-模块化策略引擎)
    *   [4.4 准入与退出规则 (Rules)](#44-准入与退出规则)
5.  [⚙️ 回测引擎与适配器 (`stock.backtest`)](#5-回测引擎与适配器)
    *   [5.1 Backtrader 适配器 (KSPStrategy)](#51-backtrader-适配器)
    *   [5.2 数据工厂 (DataFactory)](#52-数据工厂)
6.  [🛠️ 常用运维与分析脚本 (`scripts`)](#6-常用运维与分析脚本)
7.  [🛡️ 数据健康与异常处理 (`stock.utils`)](#7-数据健康与异常处理)

---

<a name="1-系统概述与核心架构"></a>
## 🚀 1. 系统概述与核心架构

KSP 策略利用高阶统计量指标捕捉市场动能异动。核心在于通过“次优区间”理论锁定最具持续性的板块与个股，并结合量比突破、波动收敛、均线乖离实现精准入场。

<a name="11-架构设计图"></a>
### 1.1 架构设计图
`数据源 (BaoStock)` -> `分钟线表 (mintues5)` -> `日线表 (daily_kline)` -> `Funnel 选股器` -> `Modular 策略引擎` -> `Backtrader 执行框架`。

<a name="12-技术栈"></a>
### 1.2 技术栈
*   **数据库**: ClickHouse (处理海量金融时序数据的高效 OLAP)
*   **数据处理**: Pandas, NumPy
*   **回测引擎**: Backtrader
*   **统计验证**: SciPy (Spearman 秩相关系数用于 IC 分析)

---

<a name="2-数据库与数据模块开发"></a>
## 2. 数据库与数据模块开发

<a name="21-clickhouse-表结构与维护"></a>
### 2.1 ClickHouse 表结构与维护
*   **存储引擎**: `ReplacingMergeTree` 引擎天然支持基于 `(code, date)` 的覆盖更新。
*   **维护注意**: 禁止在任务流中使用异步 `DELETE`。直接 `INSERT` 即可，随后调用 `optimize_table` 强制合并。
*   **新增字段**: `is_listed_180` (Int32)，用于快速判断个股是否满足 180 天上市门槛。

<a name="22-数据上下文"></a>
### 2.2 数据上下文
*   **`ConceptContext` (v2)**: 类级别缓存，提供概念与成分股的高效双向映射。
*   **`DailyContext`**: 核心聚合引擎，负责 OHLC 聚合及 POC (Point of Control) 的计算。

---

<a name="3-数据流水线任务"></a>
## 3. 数据流水线任务

数据更新是一个严格的串行流程，由 `scripts/full_daily_update.py` 统筹。

### 3.1 `Min5UpdateTask`
*   **职责**: 多进程抓取原始分钟线数据。
*   **注意**: 必须在 BaoStock 收盘后运行，若获取列表失败，检查网络连接。

### 3.2 `DailyAggregationTask`
*   **职责**: 分钟线聚合为日线基础数据。
*   **注意**: 处理冷启动问题时需清理目标时间段。

### 3.3 `RefreshFactorsTask`
*   **职责**: 计算全局截面排名。
*   **注意**: 必须强制转换排名列为 `Int32`，防止 ClickHouse 插入类型报错。

---

<a name="4-策略核心逻辑开发"></a>
## 4. 策略核心逻辑开发

<a name="41-次优区间哲学"></a>
### 4.1 次优区间哲学
避开全市场 Top 5% (D1 组) 的“拥挤交易”，聚焦于排名在 **5% - 30% (D2-D3 组)** 的标的。

<a name="42-选股漏斗"></a>
### 4.2 选股漏斗 (`stock/selector/funnel.py`)
1.  **`InitialUniverseStep`**: 过滤上市 < 180 天的个股（使用 `is_listed_180` 因子）。
2.  **`RangeConceptRankingStep`**: 锁定 KSP 均分在 [20, 100] 名的次优概念，从中选出 **Top 3**。
3.  **`FinalSelectionStep`**: 释放选定概念下的**全量成分股**，扩大策略层的样本选择空间。

<a name="43-模块化策略引擎"></a>
### 4.3 模块化策略引擎 (`stock/strategy/modular_core.py`)
*   **优先级排序**: 在 `filter_candidates` 中对满足条件的候选股进行 **5d KSP 排名升序排列**，确保资金优先分配给质量最高的次优股。

<a name="44-准入与退出规则"></a>
### 4.4 准入与退出规则 (`stock/strategy/rules.py`)
*   **准入规则**:
    *   `RangeRankEntryRule`: 5d KSP 排名处于 **[440, 1300]**。
    *   `VolumeRatioEntryRule`: 当日成交量 > 5日均值 **1.5倍**。
    *   `VolatilityConvergenceRule`: 当日振幅 < **4%**。
    *   `MovingAverageBiasRule`: 20日均线乖离率在 **[-3%, +5%]**。
*   **退出规则**:
    *   `RankExitRule`: 5d 排名跌出 **1500** 名。
    *   `BottomRankExitRule`: 5d 排名跌出 **3500** 名（强制清仓）。
    *   固定止盈 **+9.9%**，固定止损 **-2.0%**。

---

<a name="5-回测引擎与适配器"></a>
## 5. 回测引擎与适配器

<a name="51-backtrader-适配器"></a>
### 5.1 Backtrader 适配器 (`stock/backtest/ksp_strategy.py`)
*   **执行方式**: **次日开盘价成交 (Market at Open)**。
*   **数据预热**: 系统通过 `start_date` 参数支持数据预热期。在预热期间（如正式回测前 120 天），策略仅更新技术指标，不产生交易决策。这确保了 MA20、量比等指标在回测开始的第一天就是成熟的。
*   **重要修复**: 重构了买入逻辑，先统一收集所有候选股指标，再进行**统一排序过滤**，解决了优先级失效问题。
*   **机制**: 每日 `next` 开始前强制**撤销所有未成交买单**。

---

<a name="6-常用运维与分析脚本"></a>
## 6. 常用运维与分析脚本
*   `scripts/full_daily_update.py`: 一键同步数据并计算因子。
*   `scripts/latest_trading_plan.py`: 基于最新截面生成每日实盘计划。
*   `scripts/analyze_ksp_comprehensive.py`: 因子 IC 与分组收益分析。
*   `scripts/backfill_data.py`: 历史数据长周期补录脚本。

---

<a name="7-数据健康与异常处理"></a>
## 7. 数据健康与异常处理 (`stock/utils/health_check.py`)
*   **检查项**: 数据库最新日期滞后性、KSP 因子覆盖率、POC 完整性。
*   **机制**: 在回测启动前强制运行自检，异常则 Fail-Fast。
