"""
Enhanced Risk Manager - Dynamic position sizing and risk control
P0 Priority Improvement
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_portfolio_heat: float = 20.0        # Max total risk in %
    max_single_position: float = 5.0        # Max single position in %
    base_atr_multiplier: float = 1.5       # Base ATR multiplier for stops
    kelly_fraction: float = 0.25            # Fraction of Kelly to use (safety)
    min_win_rate_for_kelly: float = 0.45    # Min win rate to use Kelly
    volatility_scaling_factor: float = 2.0  # Volatility adjustment intensity


class EnhancedRiskManager:
    """
    Enhanced risk management with:
    - Kelly Criterion position sizing
    - Volatility-adjusted position sizing
    - Portfolio heat control
    - Adaptive ATR multipliers
    - Dynamic stop loss adjustment
    """

    PARAMS = RiskParameters()

    @classmethod
    def calculate_position_size(
        cls,
        confidence: float,
        risk_reward: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        atr_percentile: float = 0.5,
        market_regime: str = "NEUTRAL",
        regime_risk_adjustment: float = 1.0,
        direction: str = "LONG",
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using multiple methods.
        
        Returns dict with:
        - position_size_pct: Recommended position size
        - method: Method used (kelly, volatility_adjusted, base)
        - kelly_pct: Kelly criterion result (if applicable)
        - volatility_adjusted_pct: Volatility-adjusted result
        - risk_amount: Risk amount in % of portfolio
        """
        # Convert confidence to effective based on direction
        if direction == "SHORT":
            effective_confidence = 100 - confidence
        else:
            effective_confidence = confidence
        
        edge = max(0, effective_confidence - 50)
        
        # Method 1: Kelly Criterion (if win rate data available)
        kelly_pct = 0.0
        if win_rate and avg_win and avg_loss and win_rate >= cls.PARAMS.min_win_rate_for_kelly:
            kelly_pct = cls._kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_pct = min(kelly_pct * cls.PARAMS.kelly_fraction, cls.PARAMS.max_single_position)
        
        # Method 2: Volatility-adjusted position sizing
        vol_adjusted_pct = cls._volatility_adjusted_position_size(
            edge, risk_reward, atr_percentile
        )
        
        # Method 3: Base formula (current method)
        base_pct = edge * risk_reward * 0.05
        
        # Method 4: Regime-adjusted
        regime_adjusted_pct = base_pct * regime_risk_adjustment
        
        # Select best method
        if kelly_pct > 0:
            final_pct = kelly_pct
            method = "KELLY"
        elif abs(atr_percentile - 0.5) > 0.2:  # Significant volatility deviation
            final_pct = vol_adjusted_pct
            method = "VOLATILITY_ADJUSTED"
        else:
            final_pct = regime_adjusted_pct
            method = "REGIME_ADJUSTED"
        
        # Cap at maximum
        final_pct = min(final_pct, cls.PARAMS.max_single_position)
        final_pct = max(final_pct, 0.5)  # Minimum position size
        
        # Calculate risk amount
        risk_amount = final_pct / risk_reward if risk_reward > 0 else final_pct
        
        return {
            "position_size_pct": round(final_pct, 2),
            "method": method,
            "kelly_pct": round(kelly_pct, 2),
            "volatility_adjusted_pct": round(vol_adjusted_pct, 2),
            "regime_adjusted_pct": round(regime_adjusted_pct, 2),
            "base_pct": round(base_pct, 2),
            "risk_amount_pct": round(risk_amount, 2),
        }

    @staticmethod
    def _kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion percentage.
        
        Kelly % = (W - (1-W)) / (avg_win/avg_loss)
        
        Where:
        - W = win rate (0-1)
        - avg_win = average winning trade return
        - avg_loss = average losing trade return (positive number)
        """
        if avg_loss <= 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate - (1 - win_rate)) / win_loss_ratio
        
        return max(0, kelly * 100)  # Convert to percentage

    @staticmethod
    def _volatility_adjusted_position_size(
        edge: float,
        risk_reward: float,
        atr_percentile: float,
    ) -> float:
        """
        Calculate position size adjusted for current volatility.
        
        High volatility = smaller positions
        Low volatility = larger positions
        """
        base_size = edge * risk_reward * 0.05
        
        # Volatility adjustment factor
        # ATR percentile 0.5 = no adjustment
        # ATR percentile > 0.5 = reduce size (high vol)
        # ATR percentile < 0.5 = increase size (low vol)
        
        vol_factor = 1.0 - (atr_percentile - 0.5) * EnhancedRiskManager.PARAMS.volatility_scaling_factor
        vol_factor = max(0.5, min(1.5, vol_factor))  # Cap adjustment
        
        return base_size * vol_factor

    @classmethod
    def calculate_adaptive_stop_loss(
        cls,
        close: float,
        atr: float,
        direction: str,
        regime: str = "NEUTRAL",
        trend_strength: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate adaptive stop loss with dynamic ATR multiplier.
        
        Strong trend = wider stops (allow room for noise)
        Weak trend = tighter stops (quick exit on reversal)
        """
        # Base ATR multiplier
        base_mult = cls.PARAMS.base_atr_multiplier
        
        # Regime adjustment
        regime_mult = {
            "STRONG_BULL": 1.8,
            "BULL": 1.6,
            "NEUTRAL": 1.5,
            "CAUTIOUS": 1.4,
            "BEAR": 1.3,
            "STRONG_BEAR": 1.2,
        }.get(regime, 1.5)
        
        # Trend strength adjustment
        trend_mult = 1.0 + (trend_strength - 0.5) * 0.5  # Scale by trend strength
        
        # Final multiplier
        final_mult = base_mult * regime_mult * trend_mult
        final_mult = max(1.0, min(2.5, final_mult))  # Cap between 1.0x and 2.5x
        
        risk = atr * final_mult
        
        if direction == "LONG":
            stop_loss = close - risk
        else:
            stop_loss = close + risk
        
        return {
            "stop_loss": round(stop_loss, 4),
            "atr_multiplier": round(final_mult, 2),
            "risk_amount": round(risk, 4),
            "risk_pct": round(risk / close * 100, 2),
        }

    @classmethod
    def check_portfolio_heat(
        cls,
        current_positions: List[Dict[str, Any]],
        new_position_risk: float,
        portfolio_value: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Check if adding new position would exceed portfolio heat limits.
        
        Portfolio heat = sum of all position risks (stop distance * position size)
        """
        current_heat = 0.0
        
        for pos in current_positions:
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            size_pct = pos.get("position_size_pct", 0)
            
            if entry > 0 and stop > 0:
                risk_per_share = abs(entry - stop) / entry
                position_risk = risk_per_share * size_pct
                current_heat += position_risk
        
        total_heat = current_heat + new_position_risk
        max_heat = cls.PARAMS.max_portfolio_heat
        
        can_add = total_heat <= max_heat
        excess = total_heat - max_heat if not can_add else 0
        
        return {
            "current_heat_pct": round(current_heat, 2),
            "new_position_risk_pct": round(new_position_risk, 2),
            "total_heat_pct": round(total_heat, 2),
            "max_heat_pct": max_heat,
            "can_add_position": can_add,
            "excess_heat_pct": round(excess, 2),
            "heat_utilization": round(total_heat / max_heat * 100, 1),
        }

    @classmethod
    def calculate_trailing_stop(
        cls,
        df: pd.DataFrame,
        direction: str,
        atr: float,
        lookback: int = 20,
        trail_atr_mult: float = 1.8,
        reference_ma_period: int = 20,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic trailing stop.
        
        For LONG: Trailing stop = highest high in lookback - (trail_atr_mult * ATR)
        For SHORT: Trailing stop = lowest low in lookback + (trail_atr_mult * ATR)
        """
        if df is None or len(df) < reference_ma_period + 5:
            return {"trailing_stop": 0.0, "activated": False}
        
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].iloc[-1]
        
        if direction.upper() == "LONG":
            anchor = float(high.iloc[-lookback:].max())
            trailing_stop = anchor - trail_atr_mult * atr
            
            # Floor at reference MA
            ref_ma = high.rolling(reference_ma_period).mean().iloc[-1]
            trailing_stop = max(trailing_stop, ref_ma - 1.5 * atr)
            
            # Only activate if price is above trailing stop
            activated = close > trailing_stop
            
        else:  # SHORT
            anchor = float(low.iloc[-lookback:].min())
            trailing_stop = anchor + trail_atr_mult * atr
            
            # Ceiling at reference MA
            ref_ma = low.rolling(reference_ma_period).mean().iloc[-1]
            trailing_stop = min(trailing_stop, ref_ma + 1.5 * atr)
            
            # Only activate if price is below trailing stop
            activated = close < trailing_stop
        
        return {
            "trailing_stop": round(trailing_stop, 4),
            "activated": activated,
            "anchor": round(anchor, 4),
            "distance_from_current": round(abs(close - trailing_stop), 4),
        }

    @classmethod
    def optimize_take_profits(
        cls,
        entry: float,
        stop_loss: float,
        direction: str,
        atr: float,
        regime: str = "NEUTRAL",
    ) -> Dict[str, Any]:
        """
        Optimize take profit levels based on regime and volatility.
        
        Strong trend = more aggressive TPs (wider targets)
        Weak trend = conservative TPs (closer targets)
        """
        risk = abs(entry - stop_loss)
        
        # Regime-based TP multipliers
        tp_multipliers = {
            "STRONG_BULL": {"tp1": 1.5, "tp2": 3.0, "tp3": 5.0},
            "BULL": {"tp1": 1.5, "tp2": 2.5, "tp3": 4.0},
            "NEUTRAL": {"tp1": 1.5, "tp2": 2.0, "tp3": 3.0},
            "CAUTIOUS": {"tp1": 1.3, "tp2": 1.8, "tp3": 2.5},
            "BEAR": {"tp1": 1.3, "tp2": 1.8, "tp3": 2.5},
            "STRONG_BEAR": {"tp1": 1.2, "tp2": 1.5, "tp3": 2.0},
        }
        
        mults = tp_multipliers.get(regime, tp_multipliers["NEUTRAL"])
        
        if direction == "LONG":
            tp1 = entry + risk * mults["tp1"]
            tp2 = entry + risk * mults["tp2"]
            tp3 = entry + risk * mults["tp3"]
        else:
            tp1 = entry - risk * mults["tp1"]
            tp2 = entry - risk * mults["tp2"]
            tp3 = entry - risk * mults["tp3"]
        
        rr1 = risk / risk  # Should be 1.0
        rr2 = mults["tp2"]
        rr3 = mults["tp3"]
        
        return {
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
            "tp3": round(tp3, 4),
            "rr1": round(rr1, 2),
            "rr2": round(rr2, 2),
            "rr3": round(rr3, 2),
            "regime": regime,
        }

    @classmethod
    def assess_position_quality(
        cls,
        confidence: float,
        risk_reward: float,
        breakout_score: float,
        volume_confirmed: bool,
        htf_aligned: bool,
        smc_aligned: bool,
    ) -> Dict[str, Any]:
        """
        Assess overall position quality and provide recommendations.
        """
        quality_score = 0.0
        flags = []
        
        # Confidence score (0-30 points)
        quality_score += min(30, confidence * 0.3)
        
        # Risk/Reward (0-25 points)
        if risk_reward >= 3.0:
            quality_score += 25
            flags.append("EXCELLENT_RR")
        elif risk_reward >= 2.0:
            quality_score += 20
            flags.append("GOOD_RR")
        elif risk_reward >= 1.5:
            quality_score += 15
            flags.append("ACCEPTABLE_RR")
        else:
            quality_score += 5
            flags.append("POOR_RR")
        
        # Breakout quality (0-20 points)
        quality_score += min(20, breakout_score * 0.2)
        
        # Confluence bonuses
        if volume_confirmed:
            quality_score += 10
            flags.append("VOLUME_CONFIRMED")
        
        if htf_aligned:
            quality_score += 8
            flags.append("HTF_ALIGNED")
        
        if smc_aligned:
            quality_score += 7
            flags.append("SMC_ALIGNED")
        
        # Quality classification
        if quality_score >= 80:
            quality = "INSTITUTIONAL_GRADE"
            recommendation = "FULL_POSITION"
        elif quality_score >= 65:
            quality = "HIGH_QUALITY"
            recommendation = "STANDARD_POSITION"
        elif quality_score >= 50:
            quality = "TRADABLE"
            recommendation = "REDUCED_POSITION"
        else:
            quality = "MARGINAL"
            recommendation = "SKIP_OR_WATCH"
        
        return {
            "quality_score": round(quality_score, 1),
            "quality": quality,
            "recommendation": recommendation,
            "flags": flags,
        }


class PerformanceTracker:
    """
    Track performance metrics for strategy evaluation.
    Essential for measuring improvement impact.
    """

    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.starting_capital = 100000.0

    def add_trade(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit: float,
        entry_time: str,
        exit_time: str,
        position_size: float,
    ):
        """Record a completed trade"""
        pnl_pct = self._calculate_pnl_pct(direction, entry, exit)
        pnl_amount = pnl_pct * position_size * self.starting_capital / 100
        
        trade = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "exit": exit,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "position_size_pct": position_size,
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "holding_days": self._calculate_holding_days(entry_time, exit_time),
        }
        
        self.trades.append(trade)
        logger.info(f"Trade recorded: {symbol} {direction} PnL: {pnl_pct:.2f}%")

    def _calculate_pnl_pct(self, direction: str, entry: float, exit: float) -> float:
        """Calculate P&L percentage"""
        if direction == "LONG":
            return (exit - entry) / entry * 100
        else:
            return (entry - exit) / entry * 100

    def _calculate_holding_days(self, entry_time: str, exit_time: str) -> int:
        """Calculate holding period in days"""
        from datetime import datetime
        try:
            entry_dt = datetime.fromisoformat(entry_time)
            exit_dt = datetime.fromisoformat(exit_time)
            return (exit_dt - entry_dt).days
        except:
            return 0

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.trades:
            return {"error": "No trades recorded"}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl_pct"] > 0])
        losing_trades = len(trades_df[trades_df["pnl_pct"] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df["pnl_pct"] < 0]["pnl_pct"].mean() if losing_trades > 0 else 0
        
        total_pnl = trades_df["pnl_amount"].sum()
        total_pnl_pct = (total_pnl / self.starting_capital) * 100
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else 0
        
        # Calculate max drawdown
        self._build_equity_curve()
        max_dd = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
        returns = trades_df["pnl_pct"].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        avg_holding_days = trades_df["holding_days"].mean()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate * 100, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "total_pnl_amount": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "avg_holding_days": round(avg_holding_days, 1),
        }

    def _build_equity_curve(self):
        """Build equity curve from trades"""
        self.equity_curve = [self.starting_capital]
        for trade in self.trades:
            self.equity_curve.append(self.equity_curve[-1] + trade["pnl_amount"])

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        if not self.equity_curve:
            return 0.0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        return abs(drawdown.min())

    def get_monthly_performance(self) -> Dict[str, Any]:
        """Get monthly performance breakdown"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        trades_df["month"] = trades_df["exit_time"].dt.to_period("M")
        
        monthly = trades_df.groupby("month").agg({
            "pnl_amount": "sum",
            "pnl_pct": "sum",
            "symbol": "count",
        }).rename(columns={"symbol": "trade_count"})
        
        monthly["equity"] = self.starting_capital + monthly["pnl_amount"].cumsum()
        
        return monthly.to_dict("index")
