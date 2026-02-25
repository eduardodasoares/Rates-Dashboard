import pandas as pd
import numpy as np

from src.signals.signal_engine import SignalEngine


def generate_report(df: pd.DataFrame) -> str:
    """
    Generate a formatted daily signal report for the most recent trading day.

    Outputs:
      - Composite score and direction
      - Per-component breakdown (which of the 6 components is driving the score)
      - Macro regime and gate status
      - Key market metrics (yields, spreads, VIX, credit, real yields)
      - Last 5 days of score history
      - Historical percentile context

    Parameters
    ----------
    df : enriched DataFrame (output of FeatureEngineer.add_curve_features)

    Returns
    -------
    str — formatted report ready for printing or saving
    """
    df = df.copy()
    df["composite_score"] = SignalEngine.composite_score(df)
    df["gated_score"]     = SignalEngine.gated_score(df)

    # Sizing (using default max_dv01=100k so report shows fractions / risk_bps)
    sizing_df = SignalEngine.position_size(df)
    sizing    = sizing_df.iloc[-1]

    last      = df.iloc[-1]
    date_str  = df.index[-1].strftime("%Y-%m-%d")
    score     = float(last["composite_score"])
    gated     = float(last["gated_score"])
    macro_reg = last.get("macro_regime", "N/A")
    macro_sc  = float(last.get("macro_scalar", 1.0))
    regime    = last.get("regime", "Unknown")

    # ── Effective thresholds (carry-adjusted) ─────────────────────────────────
    t_steep = float(sizing.get("thresh_steep", 1.3))
    t_flat  = float(sizing.get("thresh_flat",  1.3))
    cr_z60  = float(last.get("carry_roll_5d_z60", 0.0)) if pd.notna(last.get("carry_roll_5d_z60")) else 0.0

    # ── Direction & implied trade ──────────────────────────────────────────────
    if score >= t_steep:
        direction, dir_arrow = "STEEPENER BIAS", "▲"
        implied_trade   = "PUT ON STEEPENER  →  long 2s10s spread (long 10Y / short 2Y)"
        trade_rationale = "Model expects 2s10s to WIDEN over next ~5 days (mean-reversion / risk-off)"
    elif score <= -t_flat:
        direction, dir_arrow = "FLATTENER BIAS", "▼"
        implied_trade   = "PUT ON FLATTENER  →  short 2s10s spread (short 10Y / long 2Y)"
        trade_rationale = "Model expects 2s10s to NARROW over next ~5 days (mean-reversion / carry)"
    else:
        direction, dir_arrow = "NEUTRAL", "─"
        implied_trade   = (f"NO TRADE  →  score below threshold  "
                           f"(need ≥ {t_steep:.2f} steep / ≥ {t_flat:.2f} flat)")
        trade_rationale = "Signal ambiguous; score must clear carry-adjusted threshold to act"

    # ── Component breakdown ───────────────────────────────────────────────────
    def _clip(val, scale=2.0):
        return float(np.clip(val / scale, -1, 1)) if pd.notna(val) else 0.0

    regime_map = {
        "Bear Steepener": -1, "Bull Steepener": -1,
        "Bear Flattener": +1, "Bull Flattener": +1,
        "Unknown": 0,
    }
    c_regime   = float(regime_map.get(regime, 0))
    c_voladj   = _clip(-last.get("2s10s_voladj", 0))
    c_vix      = _clip(last.get("VIX_z20", 0))
    c_frontend = _clip(last.get("1y2y_z20", 0))

    regime_note = {
        "Bear Steepener": "steepening momentum → expect fade",
        "Bull Steepener": "steepening momentum → expect fade",
        "Bear Flattener": "flattening momentum → expect snap-back",
        "Bull Flattener": "flattening momentum → expect snap-back",
        "Unknown":        "regime undefined",
    }.get(regime, "unknown")

    def _fmt(val):
        return f"{val:+.2f}" if pd.notna(val) else "N/A"

    voladj_val   = last.get("2s10s_voladj", np.nan)
    funding_z20  = last.get("SOFR_TBill_spread_z20", np.nan)
    vix_z20      = last.get("VIX_z20", np.nan)
    frontend_z20 = last.get("1y2y_z20", np.nan)

    # ── Historical percentile ─────────────────────────────────────────────────
    pct = (df["composite_score"] <= score).mean()
    q_cuts = df["composite_score"].quantile([0.2, 0.4, 0.6, 0.8])
    q5_thr = q_cuts[0.8]
    q1_thr = q_cuts[0.2]

    if score >= q5_thr:
        quintile_label = "IN Q5 → STEEPENER signal ACTIVE  (put on steepener)"
    elif score <= q1_thr:
        quintile_label = "IN Q1 → FLATTENER signal ACTIVE  (put on flattener)"
    elif score >= q_cuts[0.6]:
        quintile_label = "Q4 → mild steepener bias  (no trade — wait for Q5)"
    elif score <= q_cuts[0.4]:
        quintile_label = "Q2 → mild flattener bias  (no trade — wait for Q1)"
    else:
        quintile_label = "Q3 → no signal"

    # ── Consecutive days at current signal direction ──────────────────────────
    recent_dir = np.sign(df["composite_score"])
    today_dir  = np.sign(score)
    streak = 0
    for d in reversed(recent_dir.values):
        if np.sign(d) == today_dir:
            streak += 1
        else:
            break

    # ── Macro scalar narrative ────────────────────────────────────────────────
    if macro_sc == 1.0:
        scalar_note = "full signal weight"
    elif macro_sc == 0.5:
        scalar_note = "half weight — moderate policy pricing"
    else:
        scalar_note = "quarter weight — aggressive policy pricing"

    # ── Build lines ───────────────────────────────────────────────────────────
    SEP  = "=" * 62
    THIN = "─" * 62

    # ── Position sizing narrative ──────────────────────────────────────────────
    score_frac  = float(sizing["score_fraction"])
    vol_20d     = float(sizing["vol_20d_bps"])
    fwd_vol     = float(sizing["fwd_vol_bps"])
    vol_sc      = float(sizing["vol_scalar"])
    risk_b      = float(sizing["risk_bps"])
    dv01_frac   = score_frac * vol_sc  # fraction of max_dv01

    if direction == "NEUTRAL":
        sizing_note = "No trade — score below threshold. Stand aside."
        sizing_detail = ""
    else:
        sizing_note = (
            f"Size at {dv01_frac:.0%} of max DV01   "
            f"(score {score_frac:.0%} of max × vol scalar {vol_sc:.2f}×)"
        )
        sizing_detail = (
            f"At $100k max DV01:  recommended = ${dv01_frac*100_000:,.0f}/bp   "
            f"Expected risk ≈ {risk_b:.1f} bps"
        )

    lines = [
        SEP,
        f"  RATES SIGNAL REPORT   {date_str}",
        SEP,
        "",
        f"  COMPOSITE SCORE:   {score:+.2f} / 4.00   {dir_arrow}  {direction}",
        f"  Gated Score:       {gated:+.2f}   ({scalar_note})",
        f"  Macro Regime:      {macro_reg}   (scalar = {macro_sc:.2f})",
        f"  Curve Regime:      {regime}",
        f"  Signal streak:     {streak} day{'s' if streak != 1 else ''} in {direction.lower()}",
        f"  Entry thresholds:  steep ≥ {t_steep:.2f}  /  flat ≥ {t_flat:.2f}   "
        f"(carry z60: {cr_z60:+.2f}  base: 1.30)",
        "",
        THIN,
        "  IMPLIED TRADE",
        THIN,
        f"  {implied_trade}",
        f"  {trade_rationale}",
        f"  Primary horizon:   ~5 trading days (signal significant through day ~15)",
        "",
        THIN,
        "  POSITION SIZING  (vol-adjusted, target 10 bps risk per trade)",
        THIN,
        f"  2s10s vol (20d):   {vol_20d:.2f} bps/day   →   5d fwd vol ≈ {fwd_vol:.2f} bps",
        f"  Score fraction:    {score_frac:.2f}   (|score| / 4)",
        f"  Vol scalar:        {vol_sc:.2f}×   (10 bps target / {fwd_vol:.2f} bps fwd vol, clipped 0.25–2.0)",
        f"  {sizing_note}",
    ]
    if sizing_detail:
        lines.append(f"  {sizing_detail}")

    # ── Carry & roll ───────────────────────────────────────────────────────────
    carry_5d    = last.get("carry_5d",    np.nan)
    roll_5d     = last.get("roll_5d",     np.nan)
    cr_5d       = last.get("carry_roll_5d", np.nan)
    sofr_val    = last.get("SOFR",        np.nan)

    if pd.notna(cr_5d):
        # Signed for the implied trade direction
        if direction != "NEUTRAL":
            trade_sign  = +1 if direction == "STEEPENER BIAS" else -1
            carry_trade = carry_5d * trade_sign if pd.notna(carry_5d) else np.nan
            roll_trade  = roll_5d  * trade_sign if pd.notna(roll_5d)  else np.nan
            cr_trade    = cr_5d   * trade_sign
        else:
            carry_trade, roll_trade, cr_trade = carry_5d, roll_5d, cr_5d

        cr_annual   = cr_5d * 252 / 5  # annualised for the steepener base case
        cr_note     = "C+R positive → trade earns carry" if cr_5d > 0 else "C+R negative → trade costs carry"

        lines += [
            "",
            THIN,
            "  CARRY & ROLL  (DV01-neutral 2s10s, 5-day hold)",
            THIN,
        ]
        if pd.notna(sofr_val):
            lines.append(f"  {'Repo (SOFR):':<22} {sofr_val:.2f}%")
        if pd.notna(carry_5d):
            lines.append(f"  {'Carry (steepener):':<22} {carry_5d:+.3f} bps/5d")
        if pd.notna(roll_5d):
            lines.append(f"  {'Roll (steepener):':<22} {roll_5d:+.3f} bps/5d")
        lines.append(f"  {'C+R (steepener):':<22} {cr_5d:+.3f} bps/5d   ({cr_annual:+.1f} bps/yr)")
        if direction != "NEUTRAL":
            lines.append(f"  {'C+R for your trade:':<22} {cr_trade:+.3f} bps   ({cr_note})")
        lines.append("")

    lines += [
        THIN,
        "  SCORE COMPONENTS  (4 validated components, range −4 to +4)",
        THIN,
        f"  {'Component':<30} {'Contribution':>12}  Detail",
        f"  {'':<30} {'':>12}",
        f"  {'Regime (contrarian)':<30} {c_regime:>+12.2f}  {regime_note}",
        f"  {'Vol-adj (contrarian)':<30} {c_voladj:>+12.2f}  2s10s voladj = {_fmt(voladj_val)}",
        f"  {'VIX momentum':<30} {c_vix:>+12.2f}  VIX z20 = {_fmt(vix_z20)}",
        f"  {'Front-end / 1y2y':<30} {c_frontend:>+12.2f}  1y2y z20 = {_fmt(frontend_z20)}",
        f"  {'─'*30}  {'─'*12}",
        f"  {'TOTAL':<30} {score:>+12.2f}",
        "",
    ]

    # ── Key yield levels ──────────────────────────────────────────────────────
    lines += [THIN, "  YIELD LEVELS", THIN]
    for t in ["2Y", "5Y", "10Y", "30Y"]:
        val = last.get(t, np.nan)
        chg = last.get(f"{t}_chg", np.nan)
        if pd.notna(val):
            chg_str = f"   ({chg:+.1f} bps today)" if pd.notna(chg) else ""
            lines.append(f"  {t+':':<10} {val:.3f}%{chg_str}")
    lines.append("")

    # ── Curve spreads ─────────────────────────────────────────────────────────
    lines += [THIN, "  CURVE SPREADS", THIN]
    _spreads = [
        ("2s10s",  last.get("2s10s", np.nan),  last.get("2s10s_z20", np.nan),  last.get("2s10s_z60", np.nan)),
        ("5s30s",  last.get("5s30s", np.nan),  last.get("5s30s_z20", np.nan),  last.get("5s30s_z60", np.nan)),
        ("3m10y",  last.get("3m10y", np.nan),  last.get("3m10y_z20", np.nan),  None),
        ("1y2y",   last.get("1y2y", np.nan),   last.get("1y2y_z20", np.nan),   None),
    ]
    for label, val, z20, z60 in _spreads:
        if pd.notna(val):
            z60_str = f"   z60: {z60:+.2f}" if z60 is not None and pd.notna(z60) else ""
            lines.append(
                f"  {label+':':<10} {val*100:+.1f} bps   z20: {z20:+.2f}{z60_str}"
                if pd.notna(z20) else
                f"  {label+':':<10} {val*100:+.1f} bps"
            )
    lines.append("")

    # ── Fed expectations ──────────────────────────────────────────────────────
    lines += [THIN, "  FED EXPECTATIONS", THIN]
    v2y_vs_ff = last.get("2Y_vs_FF", np.nan)
    ff_val    = last.get("Fed_Funds", np.nan)
    if pd.notna(v2y_vs_ff):
        gate_note = ""
        if v2y_vs_ff > 1.50:
            gate_note = "  ← GATE ACTIVE: aggressive hike pricing"
        elif v2y_vs_ff > 1.00:
            gate_note = "  ← GATE ACTIVE: moderate hike pricing"
        elif v2y_vs_ff < -1.50:
            gate_note = "  ← GATE ACTIVE: aggressive cut pricing"
        elif v2y_vs_ff < -1.00:
            gate_note = "  ← GATE ACTIVE: moderate cut pricing"
        else:
            gate_note = "  (neutral zone — no gate)"
        lines.append(f"  {'2Y vs FF:':<14} {v2y_vs_ff:+.3f}%{gate_note}")
    if pd.notna(ff_val):
        lines.append(f"  {'Fed Funds:':<14} {ff_val:.2f}%")
    lines.append("")

    # ── Risk sentiment ────────────────────────────────────────────────────────
    lines += [THIN, "  RISK SENTIMENT", THIN]
    vix_val  = last.get("VIX", np.nan)
    vix_m5d  = last.get("VIX_mom_5d", np.nan)
    ig_val   = last.get("IG_OAS", np.nan)
    ig_z20   = last.get("IG_OAS_z20", np.nan)
    hy_val   = last.get("HY_OAS", np.nan)
    hy_z20   = last.get("HY_OAS_z20", np.nan)
    if pd.notna(vix_val):
        m5d_str = f"   5d mom: {vix_m5d:+.1f}" if pd.notna(vix_m5d) else ""
        lines.append(f"  {'VIX:':<14} {vix_val:.1f}   z20: {_fmt(vix_z20)}{m5d_str}")
    if pd.notna(ig_val):
        lines.append(f"  {'IG OAS:':<14} {ig_val:.1f} bps   z20: {_fmt(ig_z20)}")
    if pd.notna(hy_val):
        lines.append(f"  {'HY OAS:':<14} {hy_val:.1f} bps   z20: {_fmt(hy_z20)}")
    lines.append("")

    # ── Real yields & breakevens ──────────────────────────────────────────────
    tips10   = last.get("10Y_TIPS", np.nan)
    tips5    = last.get("5Y_TIPS", np.nan)
    be10     = last.get("10Y_BE", np.nan)
    be5      = last.get("5Y_BE", np.nan)
    be_slope = last.get("BE_slope", np.nan)
    if pd.notna(tips10) or pd.notna(be10):
        lines += [THIN, "  REAL YIELDS & BREAKEVENS", THIN]
        if pd.notna(tips10):
            lines.append(f"  {'10Y TIPS:':<14} {tips10:.3f}%   z20: {_fmt(last.get('10Y_TIPS_z20', np.nan))}")
        if pd.notna(tips5):
            lines.append(f"  {'5Y TIPS:':<14} {tips5:.3f}%")
        if pd.notna(be10):
            lines.append(f"  {'10Y BE:':<14} {be10:.3f}%   z20: {_fmt(last.get('10Y_BE_z20', np.nan))}   mom5d: {_fmt(last.get('10Y_BE_mom_5d', np.nan))}")
        if pd.notna(be5):
            lines.append(f"  {'5Y BE:':<14} {be5:.3f}%")
        if pd.notna(be_slope):
            lines.append(f"  {'BE Slope:':<14} {be_slope:.3f}%   (10Y BE − 5Y BE)")
        lines.append("")

    # ── Funding stress ────────────────────────────────────────────────────────
    sofr_tbill = last.get("SOFR_TBill_spread", np.nan)
    sofr_gc    = last.get("SOFR_GC_spread", np.nan)
    if pd.notna(sofr_tbill):
        lines += [THIN, "  FUNDING STRESS", THIN]
        lines.append(f"  {'SOFR−TBill:':<18} {sofr_tbill:+.1f} bps   z20: {_fmt(funding_z20)}")
        if pd.notna(sofr_gc):
            lines.append(f"  {'SOFR−GC (OBFR):':<18} {sofr_gc:+.1f} bps   z20: {_fmt(last.get('SOFR_GC_spread_z20', np.nan))}")
        lines.append("")

    # ── Signal history ────────────────────────────────────────────────────────
    lines += [
        THIN,
        "  SIGNAL HISTORY — last 5 trading days",
        THIN,
        f"  {'Date':<14} {'Score':>8}   {'Macro':>14}   Regime",
    ]
    history = df[["composite_score", "macro_regime", "regime"]].tail(5)
    for idx, row in history.iterrows():
        today_tag = "  ← TODAY" if idx == df.index[-1] else ""
        lines.append(
            f"  {idx.strftime('%Y-%m-%d'):<14} "
            f"{row['composite_score']:>+8.2f}   "
            f"{str(row.get('macro_regime', 'N/A')):>14}   "
            f"{row['regime']}{today_tag}"
        )

    # ── Historical context ────────────────────────────────────────────────────
    lines += [
        "",
        THIN,
        "  HISTORICAL CONTEXT (full sample)",
        THIN,
        f"  Current score ({score:+.2f}) is at the {pct:.0%} percentile",
        f"  Q5 threshold  (top 20%):  score > {q5_thr:+.2f}",
        f"  Q1 threshold  (bot 20%):  score < {q1_thr:+.2f}",
        f"  Signal reading: {quintile_label}",
        "",
        SEP,
    ]

    return "\n".join(lines)
