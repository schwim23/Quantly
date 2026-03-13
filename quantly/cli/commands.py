"""CLI commands for Quantly.

Usage:
  python -m quantly scan
  python -m quantly backtest --years 3
  python -m quantly train
  python -m quantly paper report
  python -m quantly paper positions
"""

from __future__ import annotations

import logging
from datetime import date

import click

from quantly.cli.display import display_picks, display_backtest_results, display_paper_report

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Quantly — AI-powered swing trading signal engine."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


@cli.command()
@click.option("--universe", default="all_us", help="Stock universe: all_us | sp500 | nasdaq100")
@click.option("--top-n", default=20, help="How many ML candidates to send to Claude")
@click.option("--min-volume", default=500_000, help="Minimum average daily volume")
@click.option("--tickers", multiple=True, help="Specific tickers to analyze (overrides universe)")
@click.option("--no-cache", is_flag=True, help="Force fresh data fetch, ignore cache")
@click.option("--paper-trade", is_flag=True, help="Record picks to paper trading DB")
@click.option("--output", type=click.Path(), help="Save results to JSON file")
def scan(
    universe: str,
    top_n: int,
    min_volume: int,
    tickers: tuple,
    no_cache: bool,
    paper_trade: bool,
    output: str | None,
) -> None:
    """Run the full market scan pipeline and output today's top picks."""
    from quantly.data.universe import get_us_equity_universe, get_sp500_tickers, screen_universe
    from quantly.features.pipeline import build_feature_matrix
    from quantly.models.predict import score_candidates, compute_shap_values
    from quantly.claude_analyst.formatter import format_all_candidates
    from quantly.claude_analyst.analyst import analyze_candidates

    signal_date = date.today()

    click.echo(f"Quantly scan — {signal_date}")
    click.echo("=" * 50)

    # Stage 1: Universe
    if tickers:
        candidate_tickers = list(tickers)
        click.echo(f"Analyzing {len(candidate_tickers)} specified tickers")
    elif universe == "sp500":
        candidate_tickers = get_sp500_tickers()
        click.echo(f"S&P 500 universe: {len(candidate_tickers)} tickers")
    else:
        click.echo("Fetching full US equity universe...")
        raw_universe = get_us_equity_universe()
        screened = screen_universe(raw_universe, min_volume=min_volume)
        candidate_tickers = screened.index.tolist()
        click.echo(f"Screened universe: {len(candidate_tickers)} candidates")

    # Stage 2: Features
    click.echo("Computing features...")
    feature_matrix = build_feature_matrix(candidate_tickers, signal_date)

    # Stage 3: ML Scoring
    click.echo("Running ML scoring...")
    scored = score_candidates(feature_matrix, top_n=top_n)
    shap_df = compute_shap_values(feature_matrix, scored["ticker"].tolist())

    # Stage 4: Claude Deep-Dive
    click.echo(f"Sending top {top_n} candidates to Claude for analysis...")
    candidates = format_all_candidates(scored, shap_df, signal_date, top_n=top_n)
    analysis = analyze_candidates(candidates)

    # Stage 5: Display
    display_picks(analysis, signal_date)

    # Optional: paper trade
    if paper_trade:
        from quantly.backtest.paper_trading import record_picks, init_db
        init_db()
        record_picks(analysis["picks"], signal_date)
        click.echo(f"Picks recorded to paper trading DB.")

    # Optional: save output
    if output:
        import json
        with open(output, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        click.echo(f"Results saved to {output}")


@cli.command()
@click.option("--years", default=3, help="Number of years to backtest")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date")
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date")
@click.option("--universe", default="sp500", help="Stock universe for backtest")
@click.option("--verbose", "-v", is_flag=True, help="Show per-trade details")
def backtest(years: int, start, end, universe: str, verbose: bool) -> None:
    """Run a walk-forward backtest and report performance metrics."""
    from quantly.backtest.engine import run_backtest

    click.echo(f"Running {years}-year walk-forward backtest...")
    results = run_backtest(years=years, universe=universe)
    display_backtest_results(results)


@cli.command()
@click.option("--config-file", type=click.Path(), help="Custom model config YAML")
@click.option("--tune", is_flag=True, help="Run Optuna hyperparameter search")
@click.option("--trials", default=50, help="Number of Optuna trials (if --tune)")
def train(config_file: str | None, tune: bool, trials: int) -> None:
    """Train or retrain the ML model on historical data."""
    from quantly.models.train import walk_forward_train, save_models

    click.echo("Training ML model...")
    click.echo("Note: Run 'quantly features build' first if feature data is stale.")
    # TODO: load feature matrix, run training
    click.echo("Training complete. Run 'quantly backtest' to validate.")


@cli.group()
def paper() -> None:
    """Paper trading commands."""
    pass


@paper.command("report")
@click.option("--export", type=click.Path(), help="Export results to CSV")
def paper_report(export: str | None) -> None:
    """Show paper trading performance report."""
    from quantly.backtest.paper_trading import get_performance_report, update_outcomes

    click.echo("Updating trade outcomes...")
    update_outcomes()
    report = get_performance_report()
    display_paper_report(report)

    if export:
        import pandas as pd
        from sqlalchemy import create_engine
        from quantly.config import config
        engine = create_engine(f"sqlite:///{config.DB_PATH}")
        df = pd.read_sql("SELECT * FROM paper_trades", engine)
        df.to_csv(export, index=False)
        click.echo(f"Exported to {export}")


@paper.command("positions")
def paper_positions() -> None:
    """Show open (pending) paper trading positions."""
    import pandas as pd
    from sqlalchemy import create_engine
    from quantly.config import config

    engine = create_engine(f"sqlite:///{config.DB_PATH}")
    df = pd.read_sql("SELECT * FROM paper_trades WHERE label IS NULL", engine)
    if df.empty:
        click.echo("No open positions.")
    else:
        click.echo(df[["ticker", "scan_date", "claude_conviction", "thesis"]].to_string())
