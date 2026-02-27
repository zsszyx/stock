import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Optional

class KSPPandasData(bt.feeds.PandasData):
    lines = ('ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'list_days', 'is_listed_180', 'poc', 'ksp_sum_5d', 'ksp_sum_10d', 'is_listed')
    params = (
        ('ksp_rank', -1), ('ksp_sum_14d_rank', -1), ('ksp_sum_10d_rank', -1), ('ksp_sum_7d_rank', -1), 
        ('ksp_sum_5d_rank', -1), ('list_days', -1), ('is_listed_180', -1), ('poc', -1), ('ksp_sum_5d', -1), ('ksp_sum_10d', -1),
        ('is_listed', -1)
    )

class KSPStrategyV3(bt.Strategy):
    params = (
        ('strategy_obj', None), ('slots', 9), ('sell_rank', 300),
        ('take_profit', 0.099), ('stop_loss', -0.02), ('ksp_period', 5), 
    )
    def __init__(self):
        self.trade_data = []
        self.last_value = 1000000.0
        
    def next(self):
        dt_datetime = datetime.combine(self.data.datetime.date(0), datetime.min.time())
        
        # 1. 退出逻辑
        for d in self.datas:
            if d._name == '_master_clock_': continue
            pos = self.getposition(d)
            if pos.size == 0: continue
            
            if len(d) > 0 and not np.isnan(d.close[0]) and d.close[0] > 0 and d.volume[0] > 0:
                price = pos.price if pos.price > 0 else d.close[0]
                profit_pct = (d.close[0] - price) / price
                current_rank = d.ksp_sum_5d_rank[0] if not np.isnan(d.ksp_sum_5d_rank[0]) else 9999
                
                if current_rank > self.p.sell_rank or profit_pct >= self.p.take_profit or profit_pct <= self.p.stop_loss:
                    self.close(data=d)
        
        # 2. 仓位检查
        active_positions = [d for d in self.datas if d._name != '_master_clock_' and self.getposition(d).size != 0]
        if len(active_positions) >= self.p.slots: return
        
        # 3. 买入逻辑
        target_codes = self.p.strategy_obj.select(dt_datetime)
        if not target_codes: return
        
        current_pos_names = [d._name for d in active_positions]
        to_buy_codes = [c for c in target_codes if c not in current_pos_names][:self.p.slots - len(active_positions)]
        
        for code in to_buy_codes:
            try:
                data = self.getdatabyname(code)
                if (len(data) > 0 and data.volume[0] > 0 and 
                    data.list_days[0] >= 180 and 
                    not np.isnan(data.poc[0]) and 
                    data.poc[0] > 0.1):
                    
                    self.order_target_percent(data=data, target=1.0/self.p.slots, exectype=bt.Order.Limit, price=data.poc[0])
            except (IndexError, KeyError, AttributeError): continue

class BTBacktester:
    def __init__(self, initial_cash: float = 1000000.0, slots: int = 9, commission: float = 0.0005):
        self.initial_cash = initial_cash
        self.slots = slots
        self.commission = commission

    def run(self, daily_df: pd.DataFrame, strategy_obj, strategy_name: str = 'concept_ksp', 
            sell_rank: int = 300, take_profit: float = 0.099, stop_loss: float = -0.02, 
            ksp_period: int = 5, benchmark_df: pd.DataFrame = None):
        print("Pre-filtering target stocks...")
        all_dates = sorted(daily_df['date'].unique())
        full_idx = pd.to_datetime(all_dates)
        
        candidate_codes = set()
        for d_str in all_dates:
            dt = datetime.strptime(d_str, '%Y-%m-%d')
            candidate_codes.update(strategy_obj.select(dt))
        
        if not candidate_codes: return None
        print(f"Total unique candidates to load: {len(candidate_codes)}")

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # 1. 注册主时钟
        if benchmark_df is not None:
            b_df = benchmark_df.copy()
            if 'date' in b_df.columns:
                b_df['datetime'] = pd.to_datetime(b_df['date'])
                b_df = b_df.set_index('datetime').reindex(full_idx).ffill().bfill()
                
                if b_df['close'].isna().any():
                    print("Warning: Benchmark has NaNs, filling with 1.0")
                    b_df['close'] = b_df['close'].fillna(1.0)
                
                # Fix: Explicitly create and add feed
                bm_feed = bt.feeds.PandasData(dataname=b_df, name='_master_clock_', plot=False)
                cerebro.adddata(bm_feed)
                self.benchmark_returns = b_df['close'].pct_change().fillna(0)
            else:
                dummy_df = pd.DataFrame(index=full_idx, data={'close': 1.0})
                cerebro.adddata(bt.feeds.PandasData(dataname=dummy_df, name='_master_clock_', plot=False))
        else:
            dummy_df = pd.DataFrame(index=full_idx, data={'close': 1.0})
            cerebro.adddata(bt.feeds.PandasData(dataname=dummy_df, name='_master_clock_', plot=False))

        # 2. 个股数据
        df_all = daily_df[daily_df['code'].isin(candidate_codes)].copy()
        df_all['datetime'] = pd.to_datetime(df_all['date'])
        df_all = df_all.set_index(['code', 'datetime']).sort_index()
        
        for code in candidate_codes:
            try:
                code_df = df_all.loc[(code, slice(None)), :].reset_index(level=0, drop=True)
                aligned_df = code_df.reindex(full_idx)
                
                fill_cols = ['open', 'high', 'low', 'close', 'poc', 'ksp_sum_5d', 'ksp_sum_5d_rank', 'list_days']
                for col in fill_cols:
                    if col in aligned_df.columns:
                        if col in ['open', 'high', 'low', 'close', 'poc']:
                            aligned_df.loc[aligned_df[col] <= 0.01, col] = np.nan
                        aligned_df[col] = aligned_df[col].ffill()
                
                aligned_df['volume'] = aligned_df['volume'].fillna(0)
                if 'list_days' not in aligned_df.columns: aligned_df['list_days'] = 0
                aligned_df['list_days'] = aligned_df['list_days'].fillna(0)
                
                data = KSPPandasData(dataname=aligned_df, name=code, plot=False)
                cerebro.adddata(data)
            except KeyError: continue

        cerebro.addstrategy(KSPStrategyV3, strategy_obj=strategy_obj, slots=self.slots, 
                            sell_rank=sell_rank, take_profit=take_profit, stop_loss=stop_loss, ksp_period=ksp_period)

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')

        print(f"Starting Backtrader execution (Final Rigorous Mode)...")
        results = cerebro.run(runonce=False)
        return results[0] # Return strategy directly, not via check

    def print_results(self, strat):
        if not strat: return
        returns = strat.analyzers.returns.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        annual = strat.analyzers.annual.get_analysis()
        
        print("\n" + "="*40 + "\n      PROFESSIONAL BACKTEST REPORT      \n" + "="*40)
        print(f"Final Value:     {strat.broker.getvalue():,.2f}")
        print(f"Total Return:    {(strat.broker.getvalue()/self.initial_cash - 1)*100:.2f}%")
        if hasattr(self, 'benchmark_returns'):
            print(f"Benchmark Ret:   {( (1 + self.benchmark_returns).prod() - 1)*100:.2f}%")
        
        ann_ret = np.mean(list(annual.values())) if annual else 0.0
        print(f"Annual Return:   {ann_ret*100:.2f}%")
        print(f"Max Drawdown:    {drawdown.max.drawdown:.2f}%")
        s_val = sharpe.get('sharperatio', 0)
        print(f"Sharpe Ratio:    {s_val if s_val is not None else 0:.2f}")
        
        if 'total' in trades:
            total = trades.total.total
            print(f"Total Trades:    {total}")
            if total > 0:
                print(f"Win Rate:        {(trades.won.total/total)*100:.2f}%")
        print("="*40)

    def plot_analysis(self, strat, save_path="backtest_analysis.png"):
        try:
            returns = pd.Series(strat.analyzers.returns.get_analysis())
            returns.index = pd.to_datetime(returns.index)
            cumulative_returns = (1 + returns).cumprod()
            ma20 = cumulative_returns.rolling(window=20).mean()
            
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
            ax1.plot(cumulative_returns, label='Strategy', color='blue', linewidth=2)
            ax1.plot(ma20, label='20d MA', color='orange', linestyle='--', alpha=0.8)
            if hasattr(self, 'benchmark_returns'):
                aligned_bench = self.benchmark_returns.reindex(returns.index).fillna(0)
                cum_bench = (1 + aligned_bench).cumprod()
                ax1.plot(cum_bench, label='Benchmark', color='red', alpha=0.6)
            
            ax1.set_title(f'Equity Curve vs Benchmark', fontsize=14)
            ax1.set_ylabel('Net Value')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e: 
            print(f"Plotting failed: {e}")
