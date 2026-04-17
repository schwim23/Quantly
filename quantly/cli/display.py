"""Rich-formatted terminal output."""

from __future__ import annotations

from datetime import date

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _conviction_color(score: int) -> str:
    if score >= 7:
        return "green"
    if score >= 4:
        return "yellow"
    return "red"


def display_picks(analysis: dict, scan_date: date) -> None:
    picks = analysis.get("picks", [])
    filtered = analysis.get("filtered_out", [])

    console.print()
    console.print(Panel(
        f"[bold]Quantly — {scan_date}[/bold]\n"
        f"[green]{len(picks)} picks[/green]  |  [dim]{len(filtered)} filtered by Claude[/dim]",
        style="bold blue",
        expand=False,
    ))

    if filtered:
        reasons = analysis.get("filtered_out_reasons", {})
        console.print("\n[dim]Filtered out:[/dim]")
        for ticker in filtered:
            reason = reasons.get(ticker, "")
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
    if "error" in results:
        console.print(f"[red]Backtest error: {results['error']}[/red]")
        return

    m = results.get("overall_metrics", {})
    table = Table(title="Backtest Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right", style="dim")

    for name, val, target in [
        ("Sharpe Ratio", f"{m.get('sharpe', 0):.2f}", "> 1.0"),
        ("Win Rate", f"{m.get('win_rate', 0):.1%}", "> 55%"),
        ("Max Drawdown", f"{m.get('max_drawdown', 0):.1%}", "< -20%"),
        ("Avg Outperformance", f"{m.get('avg_outperformance', 0):.2%}", "> +2%"),
        ("Annualized Return", f"{m.get('annualized_return', 0):.1%}", "> 15%"),
        ("Total Picks", str(int(m.get("total_picks", 0))), ""),
    ]:
        table.add_row(name, val, target)

    console.print(table)


def display_paper_report(report: dict) -> None:
    if "message" in report:
        console.print(f"[yellow]{report['message']}[/yellow]")
        return

    wr = report.get("win_rate", 0)
    console.print(Panel(
        f"[bold]Paper Trading Report[/bold]\n\n"
        f"Total completed trades: {report.get('total_trades', 0)}\n"
        f"Win rate: [{'green' if wr > 0.55 else 'red'}]{wr:.1%}[/]\n"
        f"Avg outperformance vs SPY: {report.get('avg_outperformance', 0):.2%}\n"
        f"Rolling 4-week win rate: {report.get('rolling_4w_win_rate', 0):.1%}\n\n"
        f"Best pick: [green]{report.get('best_pick', 'N/A')}[/green]  "
        f"Worst pick: [red]{report.get('worst_pick', 'N/A')}[/red]",
        title="Paper Trading",
        border_style="blue",
    ))
