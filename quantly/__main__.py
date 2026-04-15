"""Entry point: python -m quantly <command>"""
from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Quantly — swing + trend-following stock signal engine."""


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=8000, show_default=True, help="Bind port")
@click.option("--reload", is_flag=True, help="Auto-reload on file changes (dev only)")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the web dashboard (includes background scheduler)."""
    import uvicorn

    uvicorn.run("quantly.web.app:app", host=host, port=port, reload=reload)


@cli.command()
@click.option(
    "--tickers",
    multiple=True,
    metavar="TICKER",
    help="Override universe with specific tickers (repeatable)",
)
def scan(tickers: tuple[str, ...]) -> None:
    """Run the full signal pipeline and print today's picks."""
    from quantly.pipeline import run_pipeline

    picks = run_pipeline(tickers=list(tickers) if tickers else None)
    if not picks:
        click.echo("No picks generated today.")
        return

    click.echo(f"\n{'═' * 62}")
    click.echo(f"  Today's Picks  ({len(picks)} candidates)")
    click.echo(f"{'═' * 62}")
    for p in picks:
        click.echo(
            f"\n  {p['ticker']:<6}  Conviction: {p['conviction']:.0f}/100"
            f"  Entry: ${p['entry_price']:.2f}  Stop: ${p['stop_price']:.2f}"
            f"  Max hold: {p['max_hold_date']}"
        )
        click.echo(f"  Drivers: {', '.join(p['top_drivers'])}")
    click.echo("")


@cli.command()
def train() -> None:
    """Retrain the LightGBM model using walk-forward cross-validation."""
    from quantly.models.train import run_training

    click.echo("Starting walk-forward training…")
    metrics = run_training()
    click.echo(
        f"Done.  Sharpe: {metrics.get('sharpe', 0):.2f}"
        f"  Win rate: {metrics.get('win_rate', 0):.1%}"
        f"  Max drawdown: {metrics.get('max_drawdown', 0):.1%}"
    )


if __name__ == "__main__":
    cli()
