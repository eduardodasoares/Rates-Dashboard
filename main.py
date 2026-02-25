import os
import sys
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.backtests.backtest import run_backtest
from src.backtests.walk_forward import run_walk_forward, compare_walk_forward
from src.signals.signal_engine import SignalEngine
from src.reports.daily_report import generate_report
from src.tracking.signal_logger import log_signal
from src.tracking.performance_tracker import print_tracking_report

FRED_KEY = os.environ.get("FRED_API_KEY")
if not FRED_KEY:
    raise EnvironmentError("FRED_API_KEY environment variable is not set.")


def run_analysis():
    loader = DataLoader(FRED_KEY)
    df = loader.load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)

    print("=== Latest Data ===")
    print(df.tail())

    print("\n=== Backtest Results (Synthetic Returns) ===")
    for tenor in ["2Y", "5Y", "10Y", "30Y"]:
        results = run_backtest(df, yield_col=tenor)
        print(f"{tenor}: {results}")

    # ── Regime performance at both horizons ───────────────────────────────────
    for horizon in [5, 21]:
        print(f"\n=== Regime Performance — Curve (2s10s, {horizon}d) ===")
        print(SignalEngine.regime_performance(df, horizon=horizon, spread_col="2s10s").to_string())

        print(f"\n=== Regime Performance — Bond (10Y, {horizon}d) ===")
        print(SignalEngine.regime_performance(df, horizon=horizon, yield_col="10Y").to_string())

    # ── Composite score ───────────────────────────────────────────────────────
    print("\n=== Composite Score (last 10 days) ===")
    print(SignalEngine.composite_score(df).tail(10).to_string())

    # ── Score performance at both horizons ────────────────────────────────────
    for horizon in [5, 21]:
        print(f"\n=== Score Performance — Curve Target (2s10s, {horizon}d) ===")
        print(SignalEngine.score_performance(df, horizon=horizon, spread_col="2s10s").to_string())

        print(f"\n=== Score Performance — Bond Target (10Y, {horizon}d) ===")
        print(SignalEngine.score_performance(df, horizon=horizon, yield_col="10Y").to_string())

    # ── Conditional regime × z-score grid (5d primary, 21d for reference) ────
    for horizon in [5, 21]:
        print(f"\n=== Conditional Regime x Z-score Grid — Curve Target (2s10s, {horizon}d) ===")
        print(SignalEngine.conditional_regime_zscore(df, horizon=horizon, spread_col="2s10s").to_string())

    # ── Walk-forward backtest (raw vs macro-gated) ────────────────────────────
    print("\n=== Walk-Forward Backtest — Curve Target (2s10s, 5d) ===")
    cmp = compare_walk_forward(df, horizon=5, spread_col="2s10s")
    print("\n--- Year-by-Year (Raw vs Gated) ---")
    print(cmp.to_string())

    # ── Signal snapshot ───────────────────────────────────────────────────────
    print("\n=== Signal Snapshot ===")
    print(SignalEngine.snapshot(df).T.to_string())


def run_report():
    """
    Print a formatted daily signal report for the most recent trading day.
    Usage: python main.py report
    """
    loader = DataLoader(FRED_KEY)
    df = loader.load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)
    print(generate_report(df))


def run_sample_split():
    """
    Compare signal performance pre-2022 vs 2022-onwards.

    Useful for understanding whether the signal works in normal rates
    environments (pre-hiking) vs the inversion/hiking cycle environment.
    Usage: python main.py split
    """
    loader = DataLoader(FRED_KEY)
    df = loader.load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)

    cutoff_year = 2022
    df_pre  = df[df.index.year < cutoff_year]
    df_post = df[df.index.year >= cutoff_year]

    print(f"=== Sample Split Analysis (pre-{cutoff_year} vs {cutoff_year}+) ===")
    print(f"  Pre-{cutoff_year}:  {len(df_pre)} obs  ({df_pre.index[0].date()} → {df_pre.index[-1].date()})")
    print(f"  Post-{cutoff_year}: {len(df_post)} obs  ({df_post.index[0].date()} → {df_post.index[-1].date()})")
    print()

    for label, subset in [(f"Pre-{cutoff_year}", df_pre), (f"{cutoff_year}+", df_post)]:
        print(f"--- {label} ---")
        for horizon in [5, 21]:
            print(f"\n  Score Performance — 2s10s, {horizon}d horizon:")
            try:
                perf = SignalEngine.score_performance(subset, horizon=horizon, spread_col="2s10s")
                print(perf[["mean_return", "sharpe_long", "sharpe_aligned", "count"]].to_string())
            except Exception as e:
                print(f"    Error: {e}")
        print()

    print("=== Walk-Forward by Period ===")
    for label, subset in [(f"Pre-{cutoff_year}", df_pre), (f"{cutoff_year}+", df_post)]:
        print(f"\n--- Walk-Forward: {label} ---")
        try:
            result = run_walk_forward(subset, horizon=5, spread_col="2s10s")
            print(result["annual"].to_string())
            print()
            print(result["summary"].to_string())
        except ValueError as e:
            print(f"  Skipped: {e}")
        print()


def run_log():
    """
    Append today's signal to logs/signal_log.csv.
    Safe to run multiple times — skips if today is already logged.
    Usage: python main.py log
    """
    loader = DataLoader(FRED_KEY)
    df = loader.load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)
    print("=== Logging signal ===")
    log_signal(df)


def run_track():
    """
    Print live performance of all logged signals resolved so far.
    Usage: python main.py track
    """
    loader = DataLoader(FRED_KEY)
    df = loader.load_treasury_data()
    df = FeatureEngineer.add_curve_features(df)
    print_tracking_report(df)


def run_bloomberg(start_date: str = "2000-01-01"):
    """
    Pull all Bloomberg data and save to data/raw/ as CSVs.

    Run this once on the machine with the Bloomberg Terminal connected.
    The saved CSVs are then committed/copied back and used by the pipeline
    without needing a live terminal.

    Usage:
        python main.py bloomberg
        python main.py bloomberg 2010-01-01   # custom start date
    """
    from src.data.bloomberg_loader import BloombergLoader, enhance_fred_data

    print(f"Connecting to Bloomberg Terminal (localhost:8194)...")
    with BloombergLoader() as bbg:

        print("  Pulling rates data (BGCR, MOVE, swap rates)...")
        bbg_rates = bbg.load_rates_data(start_date=start_date)
        print(f"  Saved {len(bbg_rates)} rows → data/raw/bloomberg_rates.csv")

        print("  Pulling Treasury futures prices + open interest...")
        bbg_oi = bbg.load_futures_oi(start_date=start_date)
        print(f"  Saved {len(bbg_oi)} rows → data/raw/futures_oi.csv")

        print("  Pulling SOFR OIS curve (Fed path pricing)...")
        bbg_ois = bbg.load_sofr_ois_path()
        print(f"  Saved {len(bbg_ois)} rows → data/raw/sofr_ois.csv")

    print("\nBloomberg pull complete. Merging with FRED data...")
    loader = DataLoader(FRED_KEY)
    fred_df = loader.load_treasury_data()
    enhanced = enhance_fred_data(fred_df, bbg_rates)
    enhanced.to_csv("data/raw/treasury_data_enhanced.csv")
    print(f"  Saved {len(enhanced)} rows → data/raw/treasury_data_enhanced.csv")
    print("\nDone. Commit data/raw/ or copy CSVs back to your main machine.")


def run_dashboard():
    from src.dashboard.app import app
    print("Launching dashboard at http://localhost:8050")
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "bloomberg":
        start = sys.argv[2] if len(sys.argv) > 2 else "2000-01-01"
        run_bloomberg(start_date=start)
    elif cmd == "dashboard":
        run_dashboard()
    elif cmd == "report":
        run_report()
    elif cmd == "split":
        run_sample_split()
    elif cmd == "log":
        run_log()
    elif cmd == "track":
        run_track()
    else:
        run_analysis()
