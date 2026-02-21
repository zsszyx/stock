# Stock Quantitative Strategy Project

## Project Structure
*   **entrypoint.py**: Main CLI entry point for all operations.
*   **run_backtest.py**: Core backtesting logic (standalone or via CLI).
*   **stock/**: Main package containing strategy, data context, and backtest engine.
    *   `stock/selector/`: Strategy logic (ConceptSelector, KSPScoreSelector).
    *   `stock/backtest/`: Backtest engine (BTBacktester) and Strategy implementation.
    *   `stock/tasks/`: Data update tasks.
    *   `stock/data_context/`: Data abstraction layers.

## Usage

### 1. Run Backtest (Main Strategy)
```bash
# Run with default settings (2025-01-01 to now)
python entrypoint.py backtest

# Run with custom parameters
python entrypoint.py backtest --start 2024-01-01 --cash 500000 --slots 5 --top-concepts 2
```

### 2. Update Data
```bash
# Update daily K-line data
python entrypoint.py update-daily

# Update 5-minute data
python entrypoint.py update-min5 --start 2025-01-01 --end 2025-01-31

# Fetch benchmark index data
python entrypoint.py fetch-benchmark
```

### 3. Factor Management
```bash
# Incrementally update factors for recent days
python entrypoint.py update-factors --days 5

# Full refresh (use with caution)
python entrypoint.py refresh-factors
```

## Key Configuration
*   **Strategy**: `ConceptSelector` (Filters by Concept Strength + KSP Score Stability).
*   **Volatility Filter**: Uses `KSP Score` (Penalty for |5-day Ret| > 3%, preference for stable distribution).
*   **Liquidity Filter**: Minimum 50M daily turnover.
