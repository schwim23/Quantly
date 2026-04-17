"""CLI commands: scan, backtest, train, paper."""

from __future__ import annotations

import logging
from datetime import date

import click

from quantly.cli.display import console, display_backtest_results, display_paper_report, display_picks


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def cli(verbose: bool) -> None:
    """Quantly — AI-powered swing trading signal engine (free data sources)."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command()
@click.option("--universe", default="sp500", show_default=True,
              help="sp500 | nasdaq100 | all_us")
@click.option("--top-n", default=20, show_default=True,
              help="ML candidates to send to Claude")
@click.option("--min-volume", default=500_000, show_default=True)
@click.option("--tickers", multiple=True, help="Analyze specific tickers only")
@click.option("--paper-trade", is_flag=True, help="Record picks to paper trading DB")
@click.option("--output", type=click.Path(), help="Save results to JSON file")
def scan(universe, top_n, min_volume, tickers, paper_trade, output) -> None:
    """Run the full pipeline and output today's top stock picks."""
    from quantly.claude_analyst.analyst import analyze_candidates
    from quantly.claude_analyst.formatter import format_all_candidates
    from quantly.data.universe import get_universe_tickers, screen_by_liquidity
    from quantly.features.pipeline import build_feature_matrix
    from quantly.models.predict import compute_shap_values, score_candidates

    signal_date = date.today()
    console.print(f"\n[bold]Quantly scan — {signal_date}[/bold]")

    if tickers:
        candidate_tickers = list(tickers)
        console.print(f"Analyzing {len(candidate_tickers)} specified tickers")
    else:
        console.print(f"Fetching {universe} universe...")
        raw = get_universe_tickers(universe)
        console.print(f"Screening {len(raw)} tickers for liquidity (min volume {min_volume:,})...")
        candidate_tickers = screen_by_liquidity(raw, min_volume=min_volume)
        console.print(f"[green]{len(candidate_tickers)} candidates passed screening[/green]")

    console.print("Computing features (this takes a few minutes)...")
    feature_matrix = build_feature_matrix(candidate_tickers, signal_date)

    console.print("Running ML scoring...")
    scored = score_candidates(feature_matrix, top_n=top_n)
    shap_df = compute_shap_values(feature_matrix, scored["ticker"].tolist())

    console.print(f"Sending top {top_n} to Claude for analysis...")
    candidates = format_all_candidates(scored, shap_df, signal_date, top_n=top_n)
    analysis = analyze_candidates(candidates)

    display_picks(analysis, signal_date)

    if paper_trade:
        from quantly.backtest.paper_trading import init_db, record_picks
        init_db()
        record_picks(analysis["picks"], signal_date)
        console.print("[green]Picks recorded to paper trading DB.[/green]")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        console.print(f"Results saved to {output}")


@cli.command()
@click.option("--years", default=3, show_default=True)
@click.option("--universe", default="sp500", show_default=True)
def backtest(years, universe) -> None:
    """Walk-forward backtest over historical data."""
    from quantly.backtest.engine import run_backtest
    console.print(f"Running {years}-year walk-forward backtest ({universe})...")
    results = run_backtest(years=years, universe=universe)
    display_backtest_results(results)


@cli.command()
@click.option("--years", default=3, show_default=True,
              help="Years of history to use for training")
@click.option("--universe", default="sp500", show_default=True)
def train(years, universe) -> None:
    """Train the ML model on historical feature data."""
    from quantly.data.universe import get_universe_tickers, screen_by_liquidity
    from quantly.features.pipeline import build_feature_matrix
    from quantly.models.labels import compute_labels
    from quantly.models.train import walk_forward_train
    from datetime import timedelta

    console.print(f"Building features for {years}-year training set...")
    tickers = screen_by_liquidity(get_universe_tickers(universe))
    signal_date = date.today() - timedelta(days=30)  # leave 30-day buffer for labels

    feature_matrix = build_feature_matrix(tickers, signal_date)
    from datetime import date as d
    # Generate monthly signal dates going back `years` years
    signal_dates_all = []
    ticker_repeats = []
    start = date.today().replace(year=date.today().year - years)
    cur = start
    while cur < signal_date:
        ticker_repeats.extend(tickers[:50])  # sample 50 tickers per month for speed
        signal_dates_all.extend([cur] * min(50, len(tickers)))
        cur = cur + timedelta(days=30)

    console.print("Computing labels (this fetches forward prices — may take a while)...")
    labels = compute_labels(ticker_repeats, signal_dates_all)

    console.print("Training models...")
    walk_forward_train(feature_matrix, labels, train_years=years)
    console.print("[green]Training complete. Run 'quantly backtest' to validate.[/green]")


@cli.group()
def paper() -> None:
    """Paper trading commands."""


@paper.command("report")
@click.option("--export", type=click.Path(), help="Export results to CSV")
def paper_report(export) -> None:
    """Show paper trading performance report."""
    from quantly.backtest.paper_trading import get_performance_report, update_outcomes
    console.print("Updating trade outcomes...")
    update_outcomes()
    display_paper_report(get_performance_report())
    if export:
        import pandas as pd
        from sqlalchemy import create_engine
        df = pd.read_sql("SELECT * FROM paper_trades", create_engine(f"sqlite:///{config.DB_PATH}"))
        df.to_csv(export, index=False)
        console.print(f"Exported to {export}")


@paper.command("positions")
def paper_positions() -> None:
    """Show open paper trading positions."""
    import pandas as pd
    from sqlalchemy import create_engine
    from quantly.config import config
    engine = create_engine(f"sqlite:///{config.DB_PATH}")
    try:
        df = pd.read_sql("SELECT ticker, scan_date, claude_conviction, thesis FROM paper_trades WHERE label IS NULL", engine)
        console.print(df.to_string(index=False) if not df.empty else "No open positions.")
    except Exception:
        console.print("No paper trades DB found — run a scan with --paper-trade first.")


# Make config available in paper_report without import error
from quantly.config import config  # noqa: E402
