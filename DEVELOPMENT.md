# KSP 股票动能策略 - 完整开发与维护文档

这份文档旨在为 KSP (Kinetic Statistical Power) 策略的开发者、维护者和使用者提供全面、深度的技术指导。文档涵盖了从底层数据库架构到高层策略逻辑的每一个细节，并提供一键跳转链接。

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
    *   [6.1 日常运维](#61-日常运维)
    *   [6.2 策略分析](#62-策略分析)
7.  [🛡️ 数据健康与异常处理 (`stock.utils`)](#7-数据健康与异常处理)

---

<a name="1-系统概述与核心架构"></a>
## 🚀 1. 系统概述与核心架构

KSP 策略是一个基于高阶统计量（偏度、峰度计算得出的 KSP 分数）来衡量个股动能的量化系统。它通过寻找“次优”概念和个股，结合量价共振（波动收敛、量比突破），试图捕捉左侧蓄势到右侧爆发的临界点。

<a name="11-架构设计图"></a>
### 1.1 架构设计图
整个系统高度解耦，数据流向单向且清晰：
`数据源 (BaoStock)` -> `分钟线表 (mintues5)` -> `日线表 (daily_kline)` -> `Funnel 选股器` -> `Modular 策略引擎` -> `Backtrader 执行框架`。

<a name="12-技术栈"></a>
### 1.2 技术栈
*   **数据库**: ClickHouse (处理海量金融时序数据的高效 OLAP)
*   **数据处理**: Pandas, NumPy
*   **回测引擎**: Backtrader
*   **统计验证**: SciPy (Spearman 秩相关系数用于 IC 分析)

---

<a name="2-数据库与数据模块开发"></a>
## 📊 2. 数据库与数据模块开发

<a name="21-clickhouse-表结构与维护"></a>
### 2.1 ClickHouse 表结构与维护 (`stock/database/repositories/clickhouse_repo.py`)
系统采用 ClickHouse 作为核心存储，所有表均使用 `ReplacingMergeTree` 引擎。
*   **维护注意**: `ReplacingMergeTree` 允许基于 `(code, date)` 进行去重更新，因此**不要**在代码中使用异步的 `ALTER TABLE ... DELETE` 命令，直接 `INSERT` 即可，随后调用 `optimize_table` 强制合并去重。
*   **核心表**:
    *   `mintues5`: 存储 5 分钟级别原始 K 线。
    *   `daily_kline`: 存储聚合后的日线数据、POC (Point of Control) 以及各种周期的 KSP 因子和截面排名。

<a name="22-数据上下文"></a>
### 2.2 数据上下文 (`stock/data_context`)
*   **`ConceptContext` (v2)**: 负责概念板块映射。
    *   **开发细节**: 采用了类级别的 TTL 内存缓存，避免频繁查库。提供了正向 `get_stocks(concept)` 和反向 `get_concept_by_stock(code)` 的 $O(1)$ 查询。
*   **`DailyContext`**: 封装数据获取和从分钟线到日线的聚合逻辑 (`from_min5` 方法)。

---

<a name="3-数据流水线任务"></a>
## 🔄 3. 数据流水线任务

数据更新是一个严格的串行流程，由 `scripts/full_daily_update.py` 统筹。相关的具体任务类定义在 `stock/tasks/` 下。

<a name="31-min5updatetask"></a>
### 3.1 `Min5UpdateTask`: 原始数据抓取 (`stock/tasks/min5_update.py`)
*   **功能**: 多进程并发从 BaoStock 获取 5 分钟 K 线。
*   **维护指南**: 如果遇到 BaoStock 连接失败，检查网络或增加重试逻辑。该任务会自动过滤 ST 股及非沪深主板/创业板股票。

<a name="32-dailyaggregationtask"></a>
### 3.2 `DailyAggregationTask`: 日线聚合与 POC (`stock/tasks/daily_update.py`)
*   **功能**: 将 `mintues5` 数据聚合为日线，并在此阶段计算基础 KSP 分数（但不包含截面排名）。这里还会计算 **POC (筹码最密集成交价)**，它是实盘执行的重要参考锚点。
*   **维护指南**: 分块处理 (Chunking) 是为了防止内存溢出。如果新增了基于分钟线的高频特征，需在 `DailyContext.from_min5` 中实现。

<a name="33-refreshfactorstask"></a>
### 3.3 `RefreshFactorsTask`: 截面排名计算 (`stock/tasks/refresh_factors.py`)
*   **功能**: 加载全量日线，利用 `pandas.groupby.rank` 计算 `ksp_sum_5d_rank`、`10d_rank` 等**横向截面数据**。
*   **开发陷阱 (Bug Fix)**: 排名计算后往往变成 float 类型（如 123.0）。在回写 ClickHouse 之前，**必须**将定义的整形列强制转换回 `int`，否则会引发 DB 插入异常。

---

<a name="4-策略核心逻辑开发"></a>
## 🧠 4. 策略核心逻辑开发

<a name="41-次优区间哲学"></a>
### 4.1 次优区间哲学
系统的灵魂。IC 统计表明，KSP 排名 Top 5% (D1组) 的标的往往因为过热而面临杀跌风险，而排名 5%-30% (D2-D3组) 的标的动能延续性最好。因此，无论是概念还是个股，我们均采用“掐头去尾”的区间截取法。

<a name="42-选股漏斗"></a>
### 4.2 选股漏斗 (`FunnelSelector` - `stock/selector/funnel.py`)
这是一个链式过滤模式 (Pipeline Pattern)。
*   **`InitialUniverseStep`**: 过滤上市 < 180 天的次新股。
*   **`RangeConceptRankingStep` (重点)**: 
    *   计算所有概念的平均 KSP。
    *   筛选落在 `[start_rank, end_rank]` (如 20-100名) 区间的概念。
    *   从中选取 Top 3。
*   **`FinalSelectionStep`**: 将被选中概念下的**所有**存活个股送入下一层（扩大样本池，而不是在概念层就做死板截断）。

<a name="43-模块化策略引擎"></a>
### 4.3 模块化策略引擎 (`ModularKSPCore` - `stock/strategy/modular_core.py`)
隔离了具体的规则，负责执行大循环。
*   **优先级排序**: 在 `filter_candidates` 中，不仅检查个股是否符合所有买入规则，还会对符合条件的个股按照 `ksp_sum_5d_rank` 进行**升序排列**。这是保证策略能买到“最优质的次优股”的关键。

<a name="44-准入与退出规则"></a>
### 4.4 准入与退出规则 (`stock/strategy/rules.py`)
所有规则实现 `EntryRule` 或 `ExitRule` 接口。
*   **`RangeRankEntryRule`**: 次优个股准入（如排名 440 - 1300）。
*   **`VolumeRatioEntryRule`**: 量价共振之**量比突破**（当日成交量需达到 5 日均值的 1.5 倍）。
*   **`VolatilityConvergenceRule`**: 量价共振之**波动收敛**（前一日振幅 < 4%）。
*   **`RankExitRule`**: 排名劣化退出（如跌破 1500 名）。
*   **`BottomRankExitRule`**: 尾部风控（跌破 3500 名，强制清仓）。

---

<a name="5-回测引擎与适配器"></a>
## ⚙️ 5. 回测引擎与适配器

<a name="51-backtrader-适配器"></a>
### 5.1 Backtrader 适配器 (`KSPStrategy` - `stock/backtest/ksp_strategy.py`)
充当 `ModularKSPCore` 与 Backtrader 框架之间的桥梁。
*   **执行逻辑 (关键修改)**:
    *   使用 `bt.Order.Market`，即 T 日产生信号，T+1 日**以开盘价直接成交**。这比严格限价单具备更高的实盘可复制性。
    *   **每日撤单**: 在 `next()` 循环的起始处，主动调用 `self.cancel(order)` 撤销昨日未成交的买单（如因涨停未买入），确保每天的仓位决策都是最新鲜的。
*   **数据传递**: 在调用核心逻辑前，适配器会预先计算好 `vol_ratio` 等动态指标，并打包成 `stock_data_map` 传递给规则引擎。

<a name="52-数据工厂"></a>
### 5.2 数据工厂 (`BTDataFeedFactory` - `stock/backtest/data_factory.py`)
*   **对齐与填充**: 保证传入 Backtrader 的数据没有时间空洞。对于缺失的因子数据（如刚上市的几天），使用极大值（如 5000）填充排名，而不是简单的 `bfill`（防止未来函数）。

---

<a name="6-常用运维与分析脚本"></a>
## 🛠️ 6. 常用运维与分析脚本 (`scripts/`)

<a name="61-日常运维"></a>
### 6.1 日常运维
*   **`full_daily_update.py`**: **收盘后第一件事**。一键完成抓取、聚合、刷因子的全流程。
*   **`latest_trading_plan.py`**: 基于最新数据生成次日的“买入清单”和“持仓维护建议”。这是**实盘操作的指导文件**。

<a name="62-策略分析"></a>
### 6.2 策略分析
在修改策略规则前，必须用数据说话：
*   **`analyze_ksp_comprehensive.py`**: 分析不同周期 KSP 排名与未来 5 日收益率的 Spearman IC 和 Decile 分组表现，用于验证因子的有效性。
*   **`analyze_concept_ksp_correlation.py`**: 概念级别的 IC 分析，用于支撑 `RangeConceptRankingStep` 的区间设定。
*   **`analyze_coverage.py`**: 统计当前选中的候选个股集中在哪些概念上，用于人工判断行情的板块聚集度。

---

<a name="7-数据健康与异常处理"></a>
## 🛡️ 7. 数据健康与异常处理 (`stock/utils/health_check.py`)

*   **`DataHealthMonitor`**: 这是系统的“心跳检测”。
    *   **何时运行**: 在回测启动前（`run_backtest.py`）、以及数据更新任务结束时（`daily_update.py`, `refresh_factors.py`）。
    *   **检查项**: 
        1. 数据库最新日期是否落后于市场交易日。
        2. KSP 5d/10d 分数的非零覆盖率是否 > 90%。
        3. POC 数据是否完整。
    *   **机制**: 如果检测到异常，触发 Fail-Fast，直接抛出警告并阻断后续的无效回测或错误交易信号生成。