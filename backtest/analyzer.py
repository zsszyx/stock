from typing import List, Dict
import pandas as pd
import numpy as np
from .broker import Broker
from .models import Direction

class Analyzer:
    """
    Calculates performance metrics for the backtest.
    """
    def __init__(self, broker: Broker):
        self.broker = broker

    def get_trade_analysis(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all trades with calculated PnL.
        """
        if not self.broker.trades:
            return pd.DataFrame()
            
        trades_df = pd.DataFrame([
            {
                'order_id': t.order_id,
                'code': t.code,
                'direction': t.direction.value,
                'quantity': t.quantity,
                'price': t.price,
                'time': t.time,
                'commission': t.commission
            } for t in self.broker.trades
        ])
        
        return trades_df

    def _calculate_round_trip_trades(self) -> List[Dict]:
        """
        Reconstructs round-trip trades (entry to exit) using FIFO matching.
        Returns a list of dictionaries containing PnL and other stats for each completed trade.
        """
        round_trips = []
        open_positions = {}  # Format: code -> list of {'qty': signed_qty, 'price': float, 'comm': float, 'time': datetime}

        for t in self.broker.trades:
            if t.code not in open_positions:
                open_positions[t.code] = []

            # Determine signed quantity: Long is positive, Short is negative
            signed_qty = t.quantity if t.direction == Direction.LONG else -t.quantity
            remaining_qty = abs(signed_qty)
            
            queue = open_positions[t.code]
            
            # While we have incoming quantity to process
            while remaining_qty > 0:
                if not queue:
                    # Nothing to match against (New Position)
                    queue.append({
                        'qty': signed_qty if t.direction == Direction.LONG else -remaining_qty,
                        'price': t.price,
                        'comm': t.commission * (remaining_qty / t.quantity),
                        'time': t.time
                    })
                    remaining_qty = 0
                    break

                head = queue[0]
                head_qty = head['qty']
                
                # Check for Match (Opposite Signs)
                # Current Long (positive) vs Head Short (negative) OR Current Short (negative) vs Head Long (positive)
                current_sign = 1 if t.direction == Direction.LONG else -1
                head_sign = 1 if head_qty > 0 else -1
                
                if current_sign != head_sign:
                    # Closing Position
                    match_qty = min(remaining_qty, abs(head_qty))
                    
                    entry_price = head['price']
                    exit_price = t.price
                    
                    # Pro-rated commissions
                    entry_comm = head['comm'] * (match_qty / abs(head_qty))
                    exit_comm = t.commission * (match_qty / t.quantity)
                    
                    # PnL Calculation
                    if head_sign == 1: # Long Close
                        gross_pnl = (exit_price - entry_price) * match_qty
                    else: # Short Close
                        gross_pnl = (entry_price - exit_price) * match_qty
                        
                    net_pnl = gross_pnl - entry_comm - exit_comm
                    
                    round_trips.append({
                        'pnl': net_pnl,
                        'entry_time': head['time'],
                        'exit_time': t.time,
                        'duration': t.time - head['time'],
                        'return_pct': (net_pnl / (entry_price * match_qty)) * 100 if entry_price > 0 else 0
                    })
                    
                    # Update Queue Head
                    if abs(head_qty) > match_qty:
                        # Partial Close
                        if head_qty > 0:
                            head['qty'] -= match_qty
                        else:
                            head['qty'] += match_qty
                        head['comm'] -= entry_comm
                        remaining_qty = 0
                    else:
                        # Full Close of this chunk
                        queue.pop(0)
                        remaining_qty -= match_qty
                        
                else:
                    # Increasing Position (Same Sign)
                    # Create new chunk for FIFO
                    queue.append({
                        'qty': signed_qty if t.direction == Direction.LONG else -remaining_qty,
                        'price': t.price,
                        'comm': t.commission * (remaining_qty / t.quantity),
                        'time': t.time
                    })
                    remaining_qty = 0
                    
        return round_trips

    def get_performance_metrics(self) -> dict:
        """
        Calculates key performance indicators (KPIs).
        """
        if not self.broker.value_history:
            return {}
            
        # --- Basic Time-Series Metrics ---
        equity_series = pd.Series(self.broker.value_history)
        returns = equity_series.pct_change().dropna()
        
        if returns.empty:
            return {}

        total_return = (equity_series.iloc[-1] - self.broker.initial_cash) / self.broker.initial_cash
        
        start_time = self.broker.trades[0].time if self.broker.trades else None
        end_time = self.broker.trades[-1].time if self.broker.trades else None
        
        if start_time and end_time:
            days = (end_time - start_time).days
            if days > 0:
                annualized_return = (1 + total_return) ** (365 / days) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        volatility = returns.std()
        
        if volatility > 0:
            sharpe_ratio = returns.mean() / volatility
            sharpe_ratio_annualized = sharpe_ratio * np.sqrt(12000) # Assuming 5min bars
        else:
            sharpe_ratio_annualized = 0.0

        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # --- Round-Trip Trade Metrics (Win Rate, PnL Ratio) ---
        round_trips = self._calculate_round_trip_trades()
        total_round_trips = len(round_trips)
        
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_pl_ratio = 0.0
        
        if total_round_trips > 0:
            wins = [t['pnl'] for t in round_trips if t['pnl'] > 0]
            losses = [t['pnl'] for t in round_trips if t['pnl'] <= 0]
            
            win_count = len(wins)
            win_rate = (win_count / total_round_trips) * 100
            
            gross_win = sum(wins)
            gross_loss = abs(sum(losses))
            
            if gross_loss > 0:
                profit_factor = gross_win / gross_loss
            else:
                profit_factor = float('inf') if gross_win > 0 else 0
                
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            if abs(avg_loss) > 0:
                avg_pl_ratio = avg_win / abs(avg_loss)
            else:
                avg_pl_ratio = float('inf') if avg_win > 0 else 0

        return {
            'Initial Capital': self.broker.initial_cash,
            'Final Equity': equity_series.iloc[-1],
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Max Drawdown (%)': max_drawdown * 100,
            'Sharpe Ratio': sharpe_ratio_annualized,
            'Total Trades': len(self.broker.trades),
            'Total Round Trips': total_round_trips,
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Avg Win/Loss Ratio': avg_pl_ratio
        }