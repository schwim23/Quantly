"""CLI entry point — `python -m quantly` or `quantly` command."""

from quantly.cli.commands import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
