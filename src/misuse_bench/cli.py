from pathlib import Path
from typing import Annotated

import typer

from misuse_bench.aggregate import aggregate as aggregate_logs
from misuse_bench.config import DEFAULT_CONFIG, load_models
from misuse_bench.run import run_cybench

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Model variant id from the config.")],
    config: Path = DEFAULT_CONFIG,
    epochs: Annotated[int, typer.Option(min=1)] = 1,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    message_limit: Annotated[int | None, typer.Option(min=1)] = None,
    token_limit: Annotated[int | None, typer.Option(min=1)] = None,
    max_connections: Annotated[int | None, typer.Option(min=1)] = None,
    log_dir: Path = Path("logs"),
) -> None:
    """Run Cybench on one configured model variant."""
    models = load_models(config)
    if model not in models:
        raise typer.BadParameter(
            f"Unknown model {model!r}; choose from {', '.join(models)}"
        )
    run_cybench(
        models[model],
        epochs=epochs,
        limit=limit,
        message_limit=message_limit,
        token_limit=token_limit,
        max_connections=max_connections,
        log_dir=log_dir,
    )


@app.command()
def aggregate(
    config: Path = DEFAULT_CONFIG,
    k: Annotated[int, typer.Option(min=1, help="Number of attempts for pass@k.")] = 1,
    log_dir: Path = Path("logs"),
    results_dir: Path = Path("results"),
) -> None:
    """Write variant scores and base-model ceilings as CSVs."""
    try:
        aggregate_logs(
            load_models(config), k=k, log_dir=log_dir, results_dir=results_dir
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Wrote {results_dir / 'cybench_scores.csv'}")
    typer.echo(f"Wrote {results_dir / 'cybench_ceiling.csv'}")
