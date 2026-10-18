"""CLI entry points; Optuna remains an optional dependency."""

from pathlib import Path

import click


@click.group()
def experiment():
    """Run, validate, and report the inference optimization lab."""


def invoke(command, config, run_dir):
    try:
        from .workflow import execute

        result = execute(command, config, run_dir)
    except ImportError as exc:
        raise click.ClickException(
            'Install experiment dependencies with pip install "llm-optimizer[optuna]"'
        ) from exc
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote {result}")


def register(command):
    @experiment.command(command)
    @click.option(
        "--config",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--run-dir", required=True, type=click.Path(file_okay=False, path_type=Path)
    )
    def launch(config, run_dir):
        invoke(command, config, run_dir)

    return launch


run = register("run")
validate = register("validate")
report = register("report")
