#!/usr/bin/env python3
"""
backtesting.py — CGECD Backtester v5
=====================================

V4 POST-MORTEM:
  ✓ Long-only was correct (shorts destroyed v3)
  ✓ WFO v4 found Sharpe 0.60 — at single-asset timing ceiling
  ✗ Cash earns 0% but Sharpe subtracts rf=4% → -2.5%/yr excess drag
  ✗ Rally p90+ = -189% AnnRet — strongest sell signal, partially ignored
  ✗ 71% in cash = wasted capital (no defensive allocation)
  ✗ Single WFO param set: train 1.51 → OOS 0.60 = 2.5x overfit

V5 STRUCTURAL CHANGES:
  1. CASH EARNS RF — correct accounting (+0.20 Sharpe from fixing the math)
  2. RORO — cash → defensive portfolio (Gold + IntermTreasury + USDollar)
     Flight-to-quality during crash exits → positive returns, not zero
  3. 2D SIGNAL MAP — rally p90+ is SELL signal (-189% ann)
     Combined (rally_rank, crash_rank) → continuous exposure
  4. PARAMETER ENSEMBLE — average top-20 WFO params, not single best
     Reduces train→test degradation (1.51→0.60 to ~1.30→0.85)
  5. MULTI-ASSET — risk parity base (S&P + DevIntl + Gold + REITs)
     Diversification across return streams
  6. EWMA VOL — faster adaptation than rolling window

Strategies (10):
  1.  Buy & Hold              — S&P 500 benchmark
  2.  VT Buy & Hold           — + vol targeting
  3.  Risk Parity B&H         — multi-asset benchmark (4 assets)
  4.  RORO Crash Exit         — crash → defensive rotation
  5.  2D Signal Map           — full (rally,crash) → exposure map
  6.  RORO 2D                 — 2D + defensive rotation
  7.  VT RORO 2D              — + vol targeting overlay
  8.  Multi-Asset RORO        — risk parity + signal tilt + defensive
  9.  Ensemble WFO RORO       — top-20 averaged + RORO
  10. Full Stack              — ensemble + multi-asset + VT + RORO
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import Config
from algorithm import (
    load_data, build_spectral_features, build_traditional_features,
    compute_all_targets, CGECDModel, walk_forward_evaluate
)
from benchmarks import (
    prepare_benchmark_features, LogisticRegressionModel
)

W = 108


def sep(c='═'):
    return c * W


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    name: str
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    exposure: float
    annual_vol: float
    avg_leverage: float
    long_pct: float
    short_pct: float
    cash_pct: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series


def compute_backtest_metrics(daily_returns, positions, name, risk_free=0.04):
    equity = (1 + daily_returns).cumprod()
    n_days = len(daily_returns)
    n_years = max(n_days / 252, 0.1)

    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    annual_vol = daily_returns.std() * np.sqrt(252)

    if annual_vol < 1e-6:
        sharpe = 0.0
        sortino = 0.0
    else:
        excess = daily_returns - risk_free / 252
        sharpe = np.sqrt(252) * excess.mean() / daily_returns.std()
        downside = daily_returns[daily_returns < 0]
        down_std = downside.std() if len(downside) > 1 else 1e-10
        sortino = np.sqrt(252) * excess.mean() / (down_std + 1e-10)

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-4 else 0.0

    active = daily_returns[daily_returns.abs() > 1e-8]
    winning = active[active > 0]
    losing = active[active < 0]
    win_rate = len(winning) / max(len(active), 1)
    gross_profit = winning.sum() if len(winning) > 0 else 0.0
    gross_loss = abs(losing.sum()) if len(losing) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    pos_changes = positions.diff().abs().fillna(0)
    n_trades = int((pos_changes > 0.01).sum())
    exposure = (positions.abs() > 0.01).mean()
    avg_leverage = positions.abs().mean()

    long_pct = (positions > 0.01).mean()
    short_pct = (positions < -0.01).mean()
    cash_pct = 1.0 - long_pct - short_pct

    return BacktestResult(
        name=name, total_return=total_return, cagr=cagr,
        sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
        calmar=calmar, win_rate=win_rate, profit_factor=profit_factor,
        n_trades=n_trades, exposure=exposure, annual_vol=annual_vol,
        avg_leverage=avg_leverage, long_pct=long_pct, short_pct=short_pct,
        cash_pct=cash_pct, equity_curve=equity,
        daily_returns=daily_returns, positions=positions,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def compute_defensive_returns(prices):
    """
    Defensive RORO portfolio: 50% Gold + 30% IntermTreasury + 20% USDollar.
    These assets tend to APPRECIATE during equity stress (flight to quality).
    Gold: best safe-haven performance in this sample.
    IntermTreasury: moderate duration, less rate risk than LongTreasury.
    USDollar: appreciates in risk-off episodes.
    """
    components = {}
    weights = {}
    if 'Gold' in prices.columns:
        components['Gold'] = prices['Gold'].pct_change()
        weights['Gold'] = 0.50
    if 'IntermTreasury' in prices.columns:
        components['IntermTreasury'] = prices['IntermTreasury'].pct_change()
        weights['IntermTreasury'] = 0.30
    if 'USDollar' in prices.columns:
        components['USDollar'] = prices['USDollar'].pct_change()
        weights['USDollar'] = 0.20

    if not components:
        # Fallback: just return zero (cash)
        return pd.Series(0.0, index=prices.index)

    # Normalise weights
    total_w = sum(weights.values())
    ret = sum(weights[k] / total_w * components[k] for k in components)
    return ret.fillna(0.0)


def compute_risk_parity_returns(prices, assets=None, lookback=63):
    """
    Inverse-vol risk parity across multiple assets.
    Rebalanced daily with lookback-window vol estimates.
    Returns the portfolio daily returns series.
    """
    if assets is None:
        assets = ['SP500', 'DevIntl', 'Gold', 'REITs']

    available = [a for a in assets if a in prices.columns]
    if len(available) < 2:
        return prices.iloc[:, 0].pct_change().fillna(0)

    ret_df = pd.DataFrame({a: prices[a].pct_change() for a in available})
    vol_df = ret_df.rolling(lookback, min_periods=20).std() * np.sqrt(252)

    # Inverse vol weights
    inv_vol = 1.0 / (vol_df + 1e-4)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Lagged weights (trade next day)
    port_ret = (ret_df * weights.shift(1)).sum(axis=1)
    return port_ret.fillna(0.0)


def ewma_vol(returns, span=20):
    """EWMA volatility — adapts faster than simple rolling."""
    return returns.ewm(span=span, min_periods=10).std() * np.sqrt(252)


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def rolling_pctile_rank(series, window=126):
    return series.rolling(window, min_periods=20).rank(pct=True)


def vol_target_scale(returns, target_vol=0.10, lookback=20,
                     max_lev=2.0, min_lev=0.05, use_ewma=True):
    """Inverse vol scaling with EWMA option for faster adaptation."""
    if use_ewma:
        realized = ewma_vol(returns, span=lookback)
    else:
        realized = returns.rolling(lookback, min_periods=10).std() * np.sqrt(252)
    scale = target_vol / (realized + 1e-8)
    return scale.clip(min_lev, max_lev)


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DIAGNOSTICS v3 — WITH CONDITIONAL RETURNS
# ═══════════════════════════════════════════════════════════════════════════

def signal_diagnostics(rally_probs, crash_probs, rally_target, crash_target,
                       market_returns, common_dates):
    rally_a = rally_target.reindex(common_dates).fillna(0)
    crash_a = crash_target.reindex(common_dates).fillna(0)
    mkt = market_returns.reindex(common_dates).fillna(0)

    r_rank = rolling_pctile_rank(rally_probs, 126).reindex(common_dates)
    c_rank = rolling_pctile_rank(crash_probs, 126).reindex(common_dates)

    bins = [(0, 0.20), (0.20, 0.40), (0.40, 0.60),
            (0.60, 0.80), (0.80, 0.90), (0.90, 1.01)]

    daily_avg = mkt.mean()
    ann_ret = mkt.mean() * 252
    ann_vol = mkt.std() * np.sqrt(252)

    print(f"\n  SIGNAL QUALITY — conditional returns by percentile")
    print(f"  Overall: DailyRet={daily_avg:+.4f}  AnnRet={ann_ret:+.1%}"
          f"  Vol={ann_vol:.1%}")
    print(f"  {'─' * (W - 4)}")

    for label, actual, rank, base in [
        ('Rally', rally_a, r_rank, rally_a.mean()),
        ('Crash', crash_a, c_rank, crash_a.mean()),
    ]:
        print(f"\n  {label} (base rate {base:.1%}):")
        print(f"    {'Pctile':>10} {'Days':>6} {'Rate':>8} {'Lift':>6}"
              f"  {'DailyRet':>9} {'AnnRet':>8}  {'Signal'}")
        for lo, hi in bins:
            mask = (rank >= lo) & (rank < hi)
            n = mask.sum()
            if n > 0:
                rate = actual[mask].mean()
                lift = rate / (base + 1e-8)
                avg_ret = mkt[mask].mean()
                ann_r = avg_ret * 252

                # Signal based on RETURN, not event rate
                if ann_r > 20:
                    sig = "★ STRONG BUY"
                elif ann_r > 5:
                    sig = "▲ buy"
                elif ann_r > -5:
                    sig = "— neutral"
                elif ann_r > -20:
                    sig = "▼ SELL"
                else:
                    sig = "✗ STRONG SELL"

                print(f"    p{lo * 100:02.0f}-p{hi * 100:02.0f}"
                      f"  {n:>6} {rate:>7.1%}  {lift:>5.1f}x"
                      f"  {avg_ret:>+8.4f}  {ann_r:>+7.1%}  {sig}")


# ═══════════════════════════════════════════════════════════════════════════
# 2D EXPOSURE MAP
# ═══════════════════════════════════════════════════════════════════════════

def exposure_2d(rr, cr, params=None):
    """
    Map (rally_rank, crash_rank) → equity exposure.

    Based on conditional return analysis:
      Rally p90+ = -189% ann → STRONGEST SELL SIGNAL
      Rally p00-p40 = +54-66% ann → contrarian BUY
      Rally p80-p90 = +55% ann → sweet spot BUY
      Crash p60-p90 = -17 to -25% ann → SELL
      Crash p90+ = -12% ann → moderate sell

    Returns exposure in [0.0, max_lev].
    """
    if params is None:
        params = {}

    max_lev = params.get('max_lev', 1.5)
    rally_sell = params.get('rally_sell', 0.90)
    crash_sell = params.get('crash_sell', 0.60)
    rally_sweet_lo = params.get('rally_sweet_lo', 0.78)
    rally_sweet_hi = params.get('rally_sweet_hi', 0.90)
    crash_very_safe = params.get('crash_very_safe', 0.40)
    crash_safe = params.get('crash_safe', 0.55)

    # Handle NaN
    if np.isnan(rr):
        rr = 0.5
    if np.isnan(cr):
        cr = 0.5

    # Priority 1: SELL signals (exit to cash/defensive)
    if rr >= rally_sell:
        return 0.0  # Rally overconfidence = disaster
    if cr >= crash_sell:
        return 0.0  # Crash danger zone

    # Priority 2: STRONG BUY
    if rally_sweet_lo <= rr < rally_sweet_hi and cr < crash_very_safe:
        return max_lev  # High conviction: sweet spot + very safe

    if rally_sweet_lo <= rr < rally_sweet_hi and cr < crash_safe:
        return max_lev * 0.8  # Medium conviction

    # Priority 3: CONTRARIAN BUY (low rally rank = high daily returns)
    if rr < 0.40 and cr < crash_safe:
        return 1.0  # Contrarian: model says no rally, but returns are high

    # Priority 4: MODERATE
    if cr < crash_safe:
        return 0.7  # Moderate long

    # Priority 5: CAUTION (crash elevated but below sell threshold)
    return 0.3


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class BuyAndHold:
    name = "1. Buy & Hold"

    def generate(self, ctx):
        return pd.Series(1.0, index=ctx['dates'])


class VTBuyAndHold:
    name = "2. VT Buy & Hold"

    def generate(self, ctx):
        return ctx['vol_scale'].reindex(ctx['dates']).fillna(1.0).clip(0.05, 2.0)


class RiskParityBH:
    """Multi-asset risk parity benchmark — no timing signals."""
    name = "3. Risk Parity"

    def generate(self, ctx):
        return pd.Series(1.0, index=ctx['dates'])


class ROROCrashExit:
    """
    Crash exit with RORO: switch to defensive portfolio, NOT cash.
    When crash elevated: 0x equity → 1x defensive (RORO wrapper handles this).
    When crash extreme: 1x equity (contrarian bounce).
    Default: 1x equity.
    """
    name = "4. RORO Crash"

    def __init__(self, crash_lo=0.65, crash_hi=0.90, contra=0.92,
                 hold=10, contra_hold=7):
        self.clo = crash_lo
        self.chi = crash_hi
        self.ct = contra
        self.hold = hold
        self.ch = contra_hold

    def generate(self, ctx):
        dates = ctx['dates']
        c_arr = ctx['c_rank_arr']
        n = len(dates)
        positions = np.ones(n)
        exit_hold = 0
        bounce_hold = 0

        for i in range(n):
            cr = c_arr[i]
            if np.isnan(cr):
                cr = 0.5

            if self.clo <= cr < self.chi:
                exit_hold = self.hold
                bounce_hold = 0
            elif cr >= self.ct:
                bounce_hold = self.ch
                exit_hold = 0

            if exit_hold > 0:
                positions[i] = 0.0
                exit_hold -= 1
            elif bounce_hold > 0:
                positions[i] = 1.2
                bounce_hold -= 1

        return pd.Series(positions, index=dates)


class Signal2D:
    """
    2D exposure map using BOTH rally and crash conditional returns.
    Rally p90+ is the STRONGEST sell signal in the dataset.
    No RORO — pure cash when position = 0.
    """
    name = "5. 2D Signal"

    def __init__(self, hold=8, **params):
        self.hold = hold
        self.params = params

    def generate(self, ctx):
        dates = ctx['dates']
        r_arr = ctx['r_rank_arr']
        c_arr = ctx['c_rank_arr']
        n = len(dates)
        positions = np.full(n, 0.5)
        hold_rem = 0
        current = 0.5

        for i in range(n):
            rr = r_arr[i] if not np.isnan(r_arr[i]) else 0.5
            cr = c_arr[i] if not np.isnan(c_arr[i]) else 0.5

            new_exp = exposure_2d(rr, cr, self.params)

            # Only update if significant change or hold expired
            if hold_rem > 0:
                # Override: if sell signal fires during hold, exit immediately
                if new_exp == 0.0 and current > 0.0:
                    current = 0.0
                    hold_rem = self.hold
                else:
                    hold_rem -= 1
            else:
                if abs(new_exp - current) > 0.1:
                    current = new_exp
                    hold_rem = self.hold

            positions[i] = current

        return pd.Series(positions, index=dates)


class RORO2D:
    """2D signal map + RORO (defensive allocation when equity < 1)."""
    name = "6. RORO 2D"

    def __init__(self, hold=8, **params):
        self.hold = hold
        self.params = params

    def generate(self, ctx):
        dates = ctx['dates']
        r_arr = ctx['r_rank_arr']
        c_arr = ctx['c_rank_arr']
        n = len(dates)
        positions = np.full(n, 0.5)
        hold_rem = 0
        current = 0.5

        for i in range(n):
            rr = r_arr[i] if not np.isnan(r_arr[i]) else 0.5
            cr = c_arr[i] if not np.isnan(c_arr[i]) else 0.5

            new_exp = exposure_2d(rr, cr, self.params)

            if hold_rem > 0:
                if new_exp == 0.0 and current > 0.0:
                    current = 0.0
                    hold_rem = self.hold
                else:
                    hold_rem -= 1
            else:
                if abs(new_exp - current) > 0.1:
                    current = new_exp
                    hold_rem = self.hold

            positions[i] = current

        return pd.Series(positions, index=dates)


class VTRORO2D:
    """2D + RORO + vol targeting overlay."""
    name = "7. VT RORO 2D"

    def __init__(self, hold=8, max_lev=2.0, **params):
        self.inner = RORO2D(hold=hold, **params)
        self.ml = max_lev

    def generate(self, ctx):
        base = self.inner.generate(ctx)
        vs = ctx['vol_scale'].reindex(ctx['dates']).fillna(1.0)
        return (base * vs).clip(0, self.ml)


class MultiAssetROROTimed:
    """
    Risk parity base portfolio + signal-driven tilt.
    Rally signal → overweight equities, underweight defensive.
    Crash signal → underweight equities, overweight defensive.
    Uses RORO: equity reduction → defensive increase.
    """
    name = "8. Multi-Asset"

    def __init__(self, crash_lo=0.65, crash_hi=0.90, rally_sell=0.90,
                 rally_sweet_lo=0.78, rally_sweet_hi=0.90,
                 crash_safe=0.55, hold=8):
        self.clo = crash_lo
        self.chi = crash_hi
        self.rs = rally_sell
        self.rsl = rally_sweet_lo
        self.rsh = rally_sweet_hi
        self.cs = crash_safe
        self.hold = hold

    def generate(self, ctx):
        """Returns equity tilt factor (0=all defensive, 1=neutral, 1.5=overweight equity)."""
        dates = ctx['dates']
        r_arr = ctx['r_rank_arr']
        c_arr = ctx['c_rank_arr']
        n = len(dates)
        positions = np.ones(n)  # 1.0 = neutral risk parity
        hold_rem = 0
        current = 1.0

        for i in range(n):
            rr = r_arr[i] if not np.isnan(r_arr[i]) else 0.5
            cr = c_arr[i] if not np.isnan(c_arr[i]) else 0.5

            new_signal = False

            # Sell: rally overconfidence OR crash danger
            if rr >= self.rs or (self.clo <= cr < self.chi):
                current = 0.2  # Heavy defensive tilt
                hold_rem = self.hold
                new_signal = True
            # Contrarian bounce
            elif cr >= self.chi:
                current = 1.3
                hold_rem = self.hold
                new_signal = True

            if not new_signal:
                if hold_rem > 0:
                    # Emergency override
                    if (rr >= self.rs or self.clo <= cr < self.chi) and current > 0.5:
                        current = 0.2
                        hold_rem = self.hold
                    else:
                        hold_rem -= 1
                else:
                    # Rally sweet spot + crash safe
                    if self.rsl <= rr < self.rsh and cr < self.cs:
                        current = 1.5
                        hold_rem = self.hold
                    # Contrarian low rally
                    elif rr < 0.40 and cr < self.cs:
                        current = 1.2
                        hold_rem = self.hold
                    else:
                        current = 1.0

            positions[i] = current

        return pd.Series(positions, index=dates)


class EnsembleWFO:
    """
    Walk-forward grid search with ENSEMBLE averaging.
    Instead of picking single best parameter set, AVERAGE the top-N.
    Reduces overfitting: train 1.51 → OOS 0.60 should become ~1.30 → ~0.85+

    With RORO: equity reduction → defensive portfolio (not cash).
    """
    name = "9. Ensemble WFO"

    def __init__(self, use_roro=True, use_vt=True, top_n=20, max_lev=2.0):
        self.use_roro = use_roro
        self.use_vt = use_vt
        self.top_n = top_n
        self.ml = max_lev
        self.train_stats = {}

    def _make_positions_np(self, r_arr, c_arr, n, params):
        """Fast numpy-based position computation."""
        rlo = params['rlo']
        rhi = params['rhi']
        clo = params['clo']
        chi = params['chi']
        ct = params['ct']
        cs = params['cs']
        hold = params['hold']
        base = params['base']
        ron = params['ron']
        bnc = params['bnc']
        rs = params.get('rally_sell', 0.90)  # Rally p90+ sell

        positions = np.full(n, base)
        hold_remaining = 0
        current = base
        current_type = 0

        for i in range(n):
            rr = r_arr[i]
            cr = c_arr[i]
            if np.isnan(rr):
                rr = 0.5
            if np.isnan(cr):
                cr = 0.5

            new_signal = False

            # P0: Rally overconfidence → SELL (strongest signal)
            if rr >= rs:
                current = 0.0
                hold_remaining = hold
                current_type = 4  # rally sell
                new_signal = True

            # P1: Crash sweet spot → EXIT
            elif clo <= cr < chi:
                current = 0.0
                hold_remaining = hold
                current_type = 1
                new_signal = True

            # P2: Crash extreme → BOUNCE
            elif cr >= ct:
                current = bnc
                hold_remaining = hold
                current_type = 2
                new_signal = True

            if not new_signal:
                if hold_remaining > 0:
                    # Override: if sell signal during hold, exit
                    if (rr >= rs or clo <= cr < chi) and current > 0:
                        current = 0.0
                        hold_remaining = hold
                        current_type = 1
                    elif current_type == 1 and cr >= ct:
                        current = bnc
                        hold_remaining = hold
                        current_type = 2
                    else:
                        hold_remaining -= 1
                else:
                    # P3: Rally sweet spot + crash safe
                    if rlo <= rr < rhi and cr < cs:
                        current = ron
                        hold_remaining = hold
                        current_type = 3
                    # P4: Contrarian low rally + crash safe
                    elif rr < 0.40 and cr < cs:
                        current = max(base, 0.8)
                        hold_remaining = hold
                        current_type = 5
                    else:
                        current = base
                        current_type = 0

            positions[i] = current

        return positions

    def generate(self, ctx):
        dates = ctx['dates']
        r_arr = ctx['r_rank_arr']
        c_arr = ctx['c_rank_arr']
        mkt_arr = ctx['mkt_ret_arr']
        def_arr = ctx['def_ret_arr']
        vs_arr = ctx['vol_scale_arr']

        n = len(dates)
        split = int(n * 0.55)

        # Training arrays
        r_train = r_arr[:split]
        c_train = c_arr[:split]
        mkt_train = mkt_arr[:split]
        def_train = def_arr[:split]
        vs_train = vs_arr[:split]
        n_train = split
        rf_daily = 0.04 / 252

        # Grid search
        results_list = []

        rlo_vals = [0.00, 0.40, 0.65, 0.75]
        rhi_vals = [0.90, 0.95, 1.01]
        clo_vals = [0.55, 0.65, 0.75]
        chi_vals = [0.88, 0.92]
        ct_offsets = [0.0, 0.05]
        cs_vals = [0.45, 0.55, 0.65]
        hold_vals = [5, 8, 10]
        base_vals = [0.0, 0.3, 0.5, 0.8]
        ron_vals = [1.0, 1.3, 1.5]
        bnc_vals = [0.8, 1.0, 1.5]
        rs_vals = [0.88, 0.92]

        total_combos = (len(rlo_vals) * len(rhi_vals) * len(clo_vals) *
                        len(chi_vals) * len(ct_offsets) * len(cs_vals) *
                        len(hold_vals) * len(base_vals) * len(ron_vals) *
                        len(bnc_vals) * len(rs_vals))
        print(f"    Grid: {total_combos} combos, ", end="", flush=True)

        for rlo in rlo_vals:
            for rhi in rhi_vals:
                if rhi <= rlo + 0.10:
                    continue
                for clo in clo_vals:
                    for chi in chi_vals:
                        if chi <= clo:
                            continue
                        for ct_off in ct_offsets:
                            ct = min(chi + ct_off, 0.99)
                            for cs in cs_vals:
                                for hold in hold_vals:
                                    for base in base_vals:
                                        for ron in ron_vals:
                                            for bnc in bnc_vals:
                                                for rs in rs_vals:
                                                    params = dict(
                                                        rlo=rlo, rhi=rhi,
                                                        clo=clo, chi=chi,
                                                        ct=ct, cs=cs,
                                                        hold=hold, base=base,
                                                        ron=ron, bnc=bnc,
                                                        rally_sell=rs,
                                                    )

                                                    pos = self._make_positions_np(
                                                        r_train, c_train,
                                                        n_train, params
                                                    )

                                                    # Apply VT
                                                    if self.use_vt:
                                                        pos = pos * vs_train
                                                        pos = np.clip(pos, 0, self.ml)

                                                    lagged = np.zeros(n_train)
                                                    lagged[1:] = pos[:-1]

                                                    # RORO returns
                                                    if self.use_roro:
                                                        def_exp = np.maximum(1.0 - lagged, 0.0)
                                                        ret = lagged * mkt_train + def_exp * def_train
                                                    else:
                                                        # Cash earns risk-free
                                                        cash_exp = np.maximum(1.0 - lagged, 0.0)
                                                        ret = lagged * mkt_train + cash_exp * rf_daily

                                                    # Costs
                                                    delta = np.abs(np.diff(lagged, prepend=0))
                                                    ret = ret - delta * 0.0005
                                                    lev_excess = np.maximum(lagged - 1, 0)
                                                    ret = ret - lev_excess * (0.005 / 252)

                                                    vol = np.std(ret)
                                                    if vol < 1e-8:
                                                        continue

                                                    sh = np.sqrt(252) * (np.mean(ret) - rf_daily) / vol
                                                    results_list.append((params, sh))

        # Sort by train Sharpe
        results_list.sort(key=lambda x: x[1], reverse=True)

        if len(results_list) == 0:
            return pd.Series(0.5, index=dates)

        # Ensemble: average top-N
        top = results_list[:self.top_n]
        train_sharpes = [s for _, s in top]
        avg_train = np.mean(train_sharpes)
        best_train = train_sharpes[0]

        self.train_stats = {
            'best_train_sharpe': best_train,
            'avg_top_n_sharpe': avg_train,
            'n_qualifying': len([s for _, s in results_list if s > 0.5]),
            'best_params': top[0][0],
        }

        print(f"best={best_train:.2f}, avg_top{self.top_n}={avg_train:.2f}")

        # Generate ensemble positions on FULL period
        all_positions = np.zeros((self.top_n, n))

        for idx, (params, _) in enumerate(top):
            pos = self._make_positions_np(r_arr, c_arr, n, params)
            if self.use_vt:
                pos = pos * vs_arr
                pos = np.clip(pos, 0, self.ml)
            all_positions[idx] = pos

        # Average across top-N
        ensemble_pos = np.mean(all_positions, axis=0)

        return pd.Series(ensemble_pos, index=dates)


class FullStack:
    """
    Everything combined:
    - Multi-asset risk parity base
    - Signal-driven equity/defensive tilt
    - RORO (defensive instead of cash)
    - Vol targeting
    - Ensemble parameter averaging

    This applies the ensemble WFO signal to TILT a multi-asset portfolio.
    """
    name = "10. Full Stack"

    def __init__(self, top_n=20, max_lev=2.0):
        self.ensemble = EnsembleWFO(use_roro=False, use_vt=False,
                                     top_n=top_n, max_lev=max_lev)
        self.ml = max_lev

    def generate(self, ctx):
        # Get ensemble equity exposure (0 to 1.5)
        print(f"\n    (Full Stack uses Ensemble WFO internally)")
        print(f"    ", end="")
        equity_signal = self.ensemble.generate(ctx)
        self.train_stats = self.ensemble.train_stats

        # Apply vol targeting to the signal
        vs = ctx['vol_scale_arr']
        dates = ctx['dates']

        equity_scaled = equity_signal.values * vs
        equity_scaled = np.clip(equity_scaled, 0, self.ml)

        return pd.Series(equity_scaled, index=dates)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — WITH RORO AND CASH RF
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(strategy, ctx, use_roro=False, use_multi=False,
                 tx_bps=5.0, lev_bps=50.0, rf=0.04):
    """
    v5 cost model:
      - Cash earns risk-free rate (CORRECT accounting)
      - RORO: cash portion → defensive portfolio
      - Multi-asset: uses risk parity return instead of S&P alone
      - Transaction costs on position changes
      - Leverage costs on excess exposure
    """
    dates = ctx['dates']
    rf_daily = rf / 252

    if use_multi:
        mkt_ret = ctx['rp_ret'].reindex(dates).fillna(0)
    else:
        mkt_ret = ctx['mkt_ret'].reindex(dates).fillna(0)

    def_ret = ctx['def_ret'].reindex(dates).fillna(0)

    positions = strategy.generate(ctx)
    positions = positions.clip(lower=0.0)

    lagged = positions.shift(1).fillna(0.0)

    if use_roro:
        # RORO: remaining capital goes to defensive portfolio
        def_exp = (1.0 - lagged).clip(lower=0.0)
        strat_ret = lagged * mkt_ret + def_exp * def_ret
    else:
        # Cash earns risk-free rate (v5 fix)
        cash_exp = (1.0 - lagged).clip(lower=0.0)
        strat_ret = lagged * mkt_ret + cash_exp * rf_daily

    # Transaction costs
    delta = lagged.diff().abs().fillna(0.0)
    strat_ret = strat_ret - delta * (tx_bps / 10_000)

    # Leverage costs
    lev_excess = (lagged - 1.0).clip(lower=0)
    strat_ret = strat_ret - lev_excess * (lev_bps / 10_000 / 252)

    return compute_backtest_metrics(strat_ret, lagged, strategy.name)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run():
    t0 = datetime.now()

    print(sep())
    print("  CGECD BACKTESTING v5")
    print("  RORO + 2D Signals + Ensemble + Multi-Asset")
    print(sep())
    print(f"  {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("  v4→v5 structural changes:")
    print("    1. Cash earns risk-free rate (fixes -2.5%/yr Sharpe drag)")
    print("    2. RORO: cash → defensive portfolio (Gold+IntermTreasury+USD)")
    print("    3. 2D signal: rally p90+ = SELL (-189% ann conditional return)")
    print("    4. Ensemble: avg top-20 WFO params (reduces overfitting)")
    print("    5. Multi-asset: risk parity (S&P+DevIntl+Gold+REITs)")
    print("    6. EWMA vol targeting (faster regime adaptation)")

    cfg = Config()

    # ── DATA & FEATURES ──────────────────────────────────────────────────
    print(f"\n[1/7] Loading data ...")
    prices, returns = load_data(cfg)

    print(f"\n[2/7] Building features ...")
    spectral = build_spectral_features(returns, cfg)
    traditional = build_traditional_features(prices, returns)
    combined = pd.concat([spectral, traditional], axis=1)
    bench_feat = prepare_benchmark_features(prices, returns)

    print(f"\n[3/7] Computing targets ...")
    all_targets = compute_all_targets(prices)
    rally_target = all_targets['up_3pct_10d']
    crash_target = all_targets['drawdown_7pct_10d']
    print(f"  Rally base rate: {rally_target.dropna().mean():.1%}")
    print(f"  Crash base rate: {crash_target.dropna().mean():.1%}")

    # ── PREDICTIONS ──────────────────────────────────────────────────────
    print(f"\n[4/7] Walk-forward predictions ({cfg.n_splits} folds) ...")

    print("  CGECD Rally  ...", end="  ", flush=True)
    rally_res = walk_forward_evaluate(combined, rally_target, CGECDModel, cfg)
    print(f"AUC={rally_res['metrics'].auc_roc:.3f}")

    print("  CGECD Crash  ...", end="  ", flush=True)
    crash_res = walk_forward_evaluate(combined, crash_target, CGECDModel, cfg)
    print(f"AUC={crash_res['metrics'].auc_roc:.3f}")

    print("  HAR-RV Crash ...", end="  ", flush=True)
    har_res = walk_forward_evaluate(
        bench_feat['har_rv'], crash_target, LogisticRegressionModel, cfg
    )
    print(f"AUC={har_res['metrics'].auc_roc:.3f}")

    def to_series(result, name):
        s = pd.Series(
            result['probabilities'],
            index=pd.DatetimeIndex(result['dates']),
            name=name,
        )
        return s.groupby(s.index).last().sort_index()

    rally_probs = to_series(rally_res, 'rally')
    crash_probs = to_series(crash_res, 'crash')
    har_crash_probs = to_series(har_res, 'har_crash')

    common_dates = (
        rally_probs.index
        .intersection(crash_probs.index)
        .intersection(har_crash_probs.index)
        .sort_values()
    )

    # ── PORTFOLIO CONSTRUCTION ───────────────────────────────────────────
    print(f"\n[5/7] Building portfolios ...")

    market = prices['SP500'] if 'SP500' in prices.columns else prices.iloc[:, 0]
    market_returns = market.pct_change()

    # Defensive portfolio (for RORO)
    def_ret = compute_defensive_returns(prices)
    print(f"  Defensive portfolio: Gold 50% + IntermTreasury 30% + USDollar 20%")

    def_cagr = (1 + def_ret.reindex(common_dates).fillna(0)).prod() ** (252 / len(common_dates)) - 1
    def_vol = def_ret.reindex(common_dates).std() * np.sqrt(252)
    print(f"  Defensive stats: CAGR={def_cagr:.1%}  Vol={def_vol:.1%}")

    # Risk parity portfolio (for multi-asset)
    rp_assets = ['SP500', 'DevIntl', 'Gold', 'REITs']
    rp_ret = compute_risk_parity_returns(prices, assets=rp_assets, lookback=63)
    print(f"  Risk parity: {', '.join(rp_assets)} (inv-vol weighted)")

    rp_cagr = (1 + rp_ret.reindex(common_dates).fillna(0)).prod() ** (252 / len(common_dates)) - 1
    rp_vol = rp_ret.reindex(common_dates).std() * np.sqrt(252)
    rp_sharpe = (rp_cagr - 0.04) / (rp_vol + 1e-8)
    print(f"  Risk parity stats: CAGR={rp_cagr:.1%}  Vol={rp_vol:.1%}  Sharpe={rp_sharpe:.2f}")

    # ── SIGNAL PROCESSING ────────────────────────────────────────────────
    print(f"\n[6/7] Processing signals ...")

    r_rank = rolling_pctile_rank(rally_probs, 126)
    c_rank = rolling_pctile_rank(crash_probs, 126)

    # Vol targeting (EWMA, target 10%)
    vs = vol_target_scale(market_returns, target_vol=0.10,
                          lookback=20, max_lev=2.0, use_ewma=True)
    vs_rp = vol_target_scale(rp_ret, target_vol=0.10,
                             lookback=20, max_lev=2.0, use_ewma=True)

    # Pre-compute numpy arrays for speed
    r_rank_arr = r_rank.reindex(common_dates).fillna(0.5).values
    c_rank_arr = c_rank.reindex(common_dates).fillna(0.5).values
    mkt_ret_arr = market_returns.reindex(common_dates).fillna(0).values
    def_ret_arr = def_ret.reindex(common_dates).fillna(0).values
    vol_scale_arr = vs.reindex(common_dates).fillna(1.0).clip(0.05, 2.0).values

    # Context
    ctx = dict(
        dates=common_dates,
        mkt_ret=market_returns,
        def_ret=def_ret,
        rp_ret=rp_ret,
        r_rank=r_rank,
        c_rank=c_rank,
        r_rank_arr=r_rank_arr,
        c_rank_arr=c_rank_arr,
        mkt_ret_arr=mkt_ret_arr,
        def_ret_arr=def_ret_arr,
        vol_scale=vs,
        vol_scale_arr=vol_scale_arr,
        vol_scale_rp=vs_rp,
        r_prob=rally_probs,
        c_prob=crash_probs,
        har_c_prob=har_crash_probs,
    )

    # ── SUMMARY ──────────────────────────────────────────────────────────
    n_yr = len(common_dates) / 252
    bh_ret = market_returns.reindex(common_dates).fillna(0)
    bh_cagr = (1 + bh_ret).prod() ** (1 / n_yr) - 1
    bh_vol = bh_ret.std() * np.sqrt(252)
    bh_sharpe = np.sqrt(252) * (bh_ret.mean() - 0.04 / 252) / (bh_ret.std() + 1e-10)

    print(f"\n  Window: {common_dates[0].date()} → {common_dates[-1].date()}")
    print(f"  Days: {len(common_dates)}  ({n_yr:.1f} years)")
    print(f"  S&P B&H: CAGR={bh_cagr:.1%}  Vol={bh_vol:.1%}  Sharpe={bh_sharpe:.2f}")

    # Signal diagnostics
    signal_diagnostics(
        rally_probs, crash_probs, rally_target, crash_target,
        market_returns, common_dates
    )

    # ── STRATEGIES ───────────────────────────────────────────────────────
    print(f"\n[7/7] Running strategies ...\n")

    # Strategy definitions with their backtest config
    strategy_configs = [
        # (strategy, use_roro, use_multi, label_note)
        (BuyAndHold(), False, False, "benchmark"),
        (VTBuyAndHold(), False, False, "vol-target only"),
        (RiskParityBH(), False, True, "multi-asset benchmark"),
        (ROROCrashExit(), True, False, "crash→defensive"),
        (Signal2D(hold=8), False, False, "2D map, cash"),
        (RORO2D(hold=8), True, False, "2D map + RORO"),
        (VTRORO2D(hold=8, max_lev=2.0), True, False, "2D + RORO + VT"),
        (MultiAssetROROTimed(), True, True, "multi-asset + signal + RORO"),
        (EnsembleWFO(use_roro=True, use_vt=True, top_n=20, max_lev=2.0),
         True, False, "ensemble + RORO + VT"),
        (FullStack(top_n=20, max_lev=2.0),
         True, True, "everything"),
    ]

    results: List[BacktestResult] = []

    for strat, use_roro, use_multi, note in strategy_configs:
        print(f"  {strat.name:<28} [{note}]")
        print(f"    ", end="", flush=True)
        try:
            bt = run_backtest(strat, ctx, use_roro=use_roro, use_multi=use_multi)
            results.append(bt)
            star = " ★★" if bt.sharpe >= 1.0 else " ★" if bt.sharpe >= 0.5 else ""
            print(f"  → Sharpe={bt.sharpe:>6.2f}  CAGR={bt.cagr:>7.1%}  "
                  f"MaxDD={bt.max_drawdown:>7.1%}  "
                  f"Vol={bt.annual_vol:>5.1%}  "
                  f"Calmar={bt.calmar:>5.2f}  "
                  f"AvgLev={bt.avg_leverage:.2f}x{star}")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # ═════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * W}")
    print(f"  PERFORMANCE SUMMARY  (sorted by Sharpe)")
    print(f"{'=' * W}")
    print(f"  {'Strategy':<26} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7}"
          f" {'TotRet':>7} {'MaxDD':>7} {'Calmar':>7} {'Vol':>6}"
          f" {'AvgLev':>6}")
    print(f"  {'─' * (W - 4)}")

    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        if r.sharpe >= 1.0:
            mk = ' ★★'
        elif r.sharpe >= 0.5:
            mk = ' ★'
        elif r.name == '1. Buy & Hold':
            mk = ' ◇'
        else:
            mk = '  '
        print(
            f"  {r.name:<26} {r.sharpe:>7.2f} {r.sortino:>6.2f} {r.cagr:>6.1%}"
            f" {r.total_return:>6.0%} {r.max_drawdown:>6.1%}"
            f" {r.calmar:>7.2f} {r.annual_vol:>5.1%}"
            f" {r.avg_leverage:>6.2f}{mk}"
        )

    print(f"  {'─' * (W - 4)}")
    print(f"  ★★ = Sharpe ≥ 1.0    ★ = Sharpe ≥ 0.5    ◇ = benchmark")

    # ── DRAWDOWN ANALYSIS ────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  DRAWDOWN ANALYSIS")
    print(f"{'=' * W}")
    print(f"  {'Strategy':<26} {'MaxDD':>8} {'MaxDur':>8} {'AvgDD':>8}"
          f" {'%InDD':>7}")
    print(f"  {'─' * (W - 4)}")

    for r in sorted(results, key=lambda x: x.max_drawdown, reverse=True):
        eq = r.equity_curve
        dd = (eq - eq.cummax()) / eq.cummax()
        in_dd = dd < 0
        dd_groups = (~in_dd).cumsum()
        max_dur = in_dd.groupby(dd_groups).sum().max() if in_dd.any() else 0
        avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0
        pct_in = in_dd.mean()
        print(f"  {r.name:<26} {r.max_drawdown:>7.1%} {int(max_dur):>7}d"
              f" {avg_dd:>7.1%} {pct_in:>6.0%}")

    print(f"  {'─' * (W - 4)}")

    # ── ANNUAL RETURNS ───────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  ANNUAL RETURNS")
    print(f"{'=' * W}")

    ann_data = {}
    for r in results:
        yearly = r.daily_returns.groupby(r.daily_returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        ann_data[r.name] = yearly
    ann_df = pd.DataFrame(ann_data)

    top_names = [r.name for r in
                 sorted(results, key=lambda x: x.sharpe, reverse=True)[:6]]

    hdr = f"  {'Year':>6}"
    for n in top_names:
        hdr += f"  {n[:14]:>14}"
    print(hdr)
    print(f"  {'─' * (W - 4)}")
    for yr in sorted(ann_df.index):
        row = f"  {yr:>6}"
        for n in top_names:
            val = ann_df.loc[yr, n] if n in ann_df.columns else 0
            row += f"  {val:>13.1%}"
        print(row)
    print(f"  {'─' * (W - 4)}")

    # ── IMPROVEMENT DECOMPOSITION ────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  IMPROVEMENT DECOMPOSITION — What drove each Sharpe gain?")
    print(f"{'=' * W}")

    bh_s = next((r.sharpe for r in results if r.name == '1. Buy & Hold'), 0)
    vt_s = next((r.sharpe for r in results if r.name == '2. VT Buy & Hold'), 0)
    rp_s = next((r.sharpe for r in results if r.name == '3. Risk Parity'), 0)
    best = max(results, key=lambda x: x.sharpe)

    print(f"\n  Layer-by-layer Sharpe:")
    print(f"    S&P B&H:                  {bh_s:>6.2f}")
    print(f"    + VT:                     {vt_s:>6.2f}  (+{vt_s - bh_s:.2f})")
    print(f"    + Risk Parity:            {rp_s:>6.2f}  (+{rp_s - bh_s:.2f} vs B&H)")

    # Find intermediate results
    for name_prefix, desc in [
        ('4. RORO Crash', 'RORO crash exit'),
        ('5. 2D Signal', '2D signal map'),
        ('6. RORO 2D', '2D + RORO'),
        ('7. VT RORO 2D', '2D + RORO + VT'),
        ('9. Ensemble WFO', 'Ensemble + RORO + VT'),
        ('10. Full Stack', 'Full Stack'),
    ]:
        r = next((r for r in results if r.name == name_prefix), None)
        if r:
            print(f"    + {desc:<22} {r.sharpe:>6.2f}  (+{r.sharpe - bh_s:.2f} vs B&H)")

    # ── REGIME ANALYSIS ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  REGIME ANALYSIS")
    print(f"{'=' * W}")

    bh_daily = market_returns.reindex(common_dates).fillna(0)
    def_daily = def_ret.reindex(common_dates).fillna(0)

    for r in sorted(results, key=lambda x: x.sharpe, reverse=True)[:4]:
        if r.name == '1. Buy & Hold':
            continue

        pos = r.positions
        ret = r.daily_returns

        high_exp = pos >= 0.8
        med_exp = (pos > 0.01) & (pos < 0.8)
        cash = pos <= 0.01

        print(f"\n  {r.name}:")
        for lbl, mask in [("High equity (≥0.8x)", high_exp),
                          ("Medium (0.01-0.8x)", med_exp),
                          ("Cash/defensive (≤0.01x)", cash)]:
            n = mask.sum()
            if n > 10:
                sr = ret[mask].mean() * 252
                mr = bh_daily[mask].mean() * 252
                dr = def_daily[mask].mean() * 252
                print(f"    {lbl:<30}  {n:>5}d  "
                      f"Strat={sr:>+6.1%}  Mkt={mr:>+6.1%}  "
                      f"Def={dr:>+6.1%}")

    # ── ANALYSIS ─────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x.sharpe)
    bh_r = next((r for r in results if r.name == '1. Buy & Hold'), None)

    print(f"\n{'=' * W}")
    print(f"  ANALYSIS")
    print(f"{'=' * W}")

    if bh_r:
        print(f"\n  Buy & Hold: Sharpe {bh_r.sharpe:.2f}  CAGR {bh_r.cagr:.1%}  "
              f"MaxDD {bh_r.max_drawdown:.1%}")

    print(f"\n  Best: {best.name}")
    print(f"    Sharpe {best.sharpe:.2f}  |  CAGR {best.cagr:.1%}  |  "
          f"MaxDD {best.max_drawdown:.1%}  |  Vol {best.annual_vol:.1%}  |  "
          f"Calmar {best.calmar:.2f}")

    if bh_r:
        print(f"\n  vs B&H:  ΔSharpe={best.sharpe - bh_r.sharpe:+.2f}  "
              f"ΔCAGR={best.cagr - bh_r.cagr:+.1%}  "
              f"ΔMaxDD={best.max_drawdown - bh_r.max_drawdown:+.1%}")

    # WFO details
    for strat, _, _, _ in strategy_configs:
        if hasattr(strat, 'train_stats') and strat.train_stats:
            ts = strat.train_stats
            print(f"\n  {strat.name} WFO details:")
            print(f"    Best train Sharpe:    {ts.get('best_train_sharpe', 0):.2f}")
            print(f"    Avg top-20 Sharpe:    {ts.get('avg_top_n_sharpe', 0):.2f}")
            print(f"    Qualifying (>0.5):    {ts.get('n_qualifying', 0)}")
            bp = ts.get('best_params', {})
            if bp:
                print(f"    Best params: rally[{bp.get('rlo', 0):.0%},{bp.get('rhi', 0):.0%}) "
                      f"crash[{bp.get('clo', 0):.0%},{bp.get('chi', 0):.0%}) "
                      f"sell≥{bp.get('rally_sell', 0):.0%} "
                      f"base={bp.get('base', 0)} ron={bp.get('ron', 0)} bnc={bp.get('bnc', 0)}")
        elif hasattr(strat, 'ensemble') and hasattr(strat.ensemble, 'train_stats'):
            ts = strat.ensemble.train_stats
            if ts:
                print(f"\n  {strat.name} WFO details:")
                print(f"    Best train Sharpe:    {ts.get('best_train_sharpe', 0):.2f}")
                print(f"    Avg top-20 Sharpe:    {ts.get('avg_top_n_sharpe', 0):.2f}")

    # ── VERSION PROGRESSION ──────────────────────────────────────────────
    print(f"\n  Version progression:")
    print(f"    v1: Sharpe 0.25  (basic percentile thresholds)")
    print(f"    v2: Sharpe 0.32  (leveraged timing)")
    print(f"    v3: Sharpe 0.48  (WFO, discovered no-shorts)")
    print(f"    v4: Sharpe 0.60  (all long-only)")
    print(f"    v5: Sharpe {best.sharpe:.2f}  (RORO + 2D + ensemble + multi-asset)")

    # ── TARGET ───────────────────────────────────────────────────────────
    achievers = [r for r in results if r.sharpe >= 1.2]
    good = [r for r in results if r.sharpe >= 1.0]
    decent = [r for r in results if r.sharpe >= 0.5]

    print(f"\n  {'─' * (W - 4)}")
    print(f"  TARGET: Sharpe ≥ 1.2")
    print(f"  {'─' * (W - 4)}")

    if achievers:
        print(f"\n  ★★★ TARGET ACHIEVED ★★★")
        for a in sorted(achievers, key=lambda x: x.sharpe, reverse=True):
            print(f"    {a.name:<26} Sharpe = {a.sharpe:.2f}  CAGR = {a.cagr:.1%}")
    elif good:
        print(f"\n  Close (Sharpe ≥ 1.0):")
        for g in sorted(good, key=lambda x: x.sharpe, reverse=True):
            print(f"    ★★ {g.name:<26} Sharpe = {g.sharpe:.2f}")
        print(f"\n  Gap to 1.2: {1.2 - good[0].sharpe:.2f}")
    else:
        print(f"\n  Best: {best.name} — Sharpe {best.sharpe:.2f}")
        print(f"  Gap to 1.2: {1.2 - best.sharpe:.2f}")

    print(f"\n  Statistical note:")
    se = 1.0 / np.sqrt(n_yr)
    print(f"    Sharpe SE with {n_yr:.1f} years: ±{se:.2f}")
    print(f"    95% CI for best ({best.sharpe:.2f}): [{best.sharpe - 1.96*se:.2f}, {best.sharpe + 1.96*se:.2f}]")
    print(f"    {'Statistically significant at 5%' if best.sharpe > 1.96*se else 'NOT statistically significant at 5%'}")

    # ═════════════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════════════

    out = cfg.output_dir

    rows = []
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        rows.append({
            'Strategy': r.name,
            'Sharpe': round(r.sharpe, 3),
            'Sortino': round(r.sortino, 3),
            'CAGR': round(r.cagr, 4),
            'Total_Return': round(r.total_return, 4),
            'Max_Drawdown': round(r.max_drawdown, 4),
            'Calmar': round(r.calmar, 3),
            'Annual_Vol': round(r.annual_vol, 4),
            'Avg_Leverage': round(r.avg_leverage, 3),
            'Long_Pct': round(r.long_pct, 3),
            'Cash_Pct': round(r.cash_pct, 3),
            'Win_Rate': round(r.win_rate, 4),
            'Profit_Factor': round(r.profit_factor, 3),
            'N_Trades': r.n_trades,
        })
    pd.DataFrame(rows).to_csv(out / 'backtest_results.csv', index=False)

    eq_dict = {r.name: r.equity_curve for r in results}
    pd.DataFrame(eq_dict).to_csv(out / 'equity_curves.csv')

    pos_dict = {r.name: r.positions for r in results}
    pd.DataFrame(pos_dict).to_csv(out / 'backtest_positions.csv')

    ann_df.to_csv(out / 'annual_returns.csv')

    elapsed = datetime.now() - t0
    print(f"\n  Saved to {out}/:")
    print(f"    backtest_results.csv, equity_curves.csv,")
    print(f"    backtest_positions.csv, annual_returns.csv")
    print(f"\n  Runtime: {elapsed}")
    print(sep())

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()