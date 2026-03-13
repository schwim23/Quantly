"""Rich-formatted terminal output for scan results and reports."""

from __future__ import annotations

from datetime import date

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

CONVICTION_COLORS = {
    range(1, 4): "red",
    range(4, 7): "yellow",
    range(7, 11): "green",
}


def _conviction_color(score: int) -> str:
    for r, color in CONVICTION_COLORS.items():
        if score in r:
            return color
    return "white"


def display_picks(analysis: dict, scan_date: date) -> None:
    """Display Claude's final pick list in a rich-formatted table."""
    picks = analysis.get("picks", [])
    filtered = analysis.get("filtered_out", [])

    console.print()
    console.print(Panel(
        f"[bold]Quantly Scan Results — {scan_date}[/bold]\n"
        f"{len(picks)} picks  |  {len(filtered)} filtered out by Claude",
        style="bold blue",
    ))

    if filtered:
        console.print(f"\n[dim]Filtered out by Claude: {', '.join(filtered)}[/dim]")
        reasons = analysis.get("filtered_out_reasons", {})
        for ticker, reason in reasons.items():
            console.print(f"  [dim]• {ticker}: {reason}[/dim]")

    console.print()

    for i, pick in enumerate(picks, 1):
        conviction = pick.get("conviction", 0)
        color = _conviction_color(conviction)

        console.print(Panel(
            f"[bold]{i}. {pick['ticker']}[/bold]  "
            f"Conviction: [{color}]{conviction}/10[/{color}]\n\n"
            f"[bold]Thesis:[/bold] {pick.get('thesis', '')}\n\n"
            f"[bold]Key signals:[/bold] {', '.join(pick.get('key_signals', []))}\n\n"
            f"[bold][red]Risk:[/red][/bold] {pick.get('biggest_risk', '')}",
            border_style=color,
        ))

    console.print()


def display_backtest_results(results: dict) -> None:
    """Display backtest performance metrics."""
    if "error" in results:
        console.print(f"[red]Backtest error: {results['error']}[/red]")
        return

    metrics = results.get("overall_metrics", {})

    table = Table(title="Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right", style="dim")

    rows = [
        ("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}", "> 1.0"),
        ("Win Rate", f"{metrics.get('win_rate', 0):.1%}", "> 55%"),
        ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}", "< -20%"),
        ("Avg Outperformance", f"{metrics.get('avg_outperformance', 0):.2%}", "> +2%"),
        ("Annualized Return", f"{metrics.get('annualized_return', 0):.1%}", "> 15%"),
        ("Total Picks", str(int(metrics.get("total_picks", 0))), ""),
    ]

    for name, value, target in rows:
        table.add_row(name, value, target)

    console.print(table)


def display_paper_report(report: dict) -> None:
    """Display paper trading performance report."""
    if "message" in report:
        console.print(f"[yellow]{report['message']}[/yellow]")
        return

    console.print(Panel(
        f"[bold]Paper Trading Performance Report[/bold]\n\n"
        f"Total trades completed: {report.get('total_trades', 0)}\n"
        f"Win rate: [{'green' if report.get('win_rate', 0) > 0.55 else 'red'}]"
        f"{report.get('win_rate', 0):.1%}[/]\n"
        f"Avg outperformance vs SPY: {report.get('avg_outperformance', 0):.2%}\n"
        f"Rolling 4-week win rate: {report.get('rolling_4w_win_rate', 0):.1%}\n\n"
        f"Best pick: [green]{report.get('best_pick', 'N/A')}[/green]  "
        f"Worst pick: [red]{report.get('worst_pick', 'N/A')}[/red]",
        title="Paper Trading",
        border_style="blue",
    ))
