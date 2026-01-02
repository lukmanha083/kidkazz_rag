"""Configuration management commands."""

from pathlib import Path

import typer

from ..config import CLIConfig
from ..output import (
    console,
    print_config,
    print_error,
    print_json,
    print_success,
    print_warning,
)
from ..utils import get_project_config_path, get_user_config_path

app = typer.Typer(help="Configuration management")


@app.command("show")
def show_config(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
    ),
) -> None:
    """Display current configuration."""
    config = CLIConfig.load()

    if json_output:
        data = config.to_dict()
        data["_sources"] = config._sources
        print_json(data)
    else:
        print_config(config.to_dict(), config._sources)

        # Show config file locations
        console.print("\n[bold cyan]Config Files[/bold cyan]")
        project_path = get_project_config_path()
        user_path = get_user_config_path()

        if project_path.exists():
            console.print(f"  Project: {project_path} [green](exists)[/green]")
        else:
            console.print(f"  Project: {project_path} [dim](not found)[/dim]")

        if user_path.exists():
            console.print(f"  User: {user_path} [green](exists)[/green]")
        else:
            console.print(f"  User: {user_path} [dim](not found)[/dim]")


@app.command("set")
def set_config(
    key: str = typer.Argument(
        ...,
        help="Configuration key (e.g., store_type, model_name)",
    ),
    value: str = typer.Argument(
        ...,
        help="Configuration value",
    ),
    project: bool = typer.Option(
        False,
        "--project",
        "-p",
        help="Set in project config (.kidkazz.toml)",
    ),
    user: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Set in user config (~/.config/kidkazz/)",
    ),
) -> None:
    """Set a configuration value."""
    # Determine which config to modify
    if project and user:
        print_error("Cannot specify both --project and --global")
        raise typer.Exit(1)

    # Load existing config
    config = CLIConfig.load()

    # Validate and set the value
    if not config.set_value(key, value):
        valid_keys = [
            "store_type", "helix_port", "helix_local",
            "embedder_type", "model_name",
            "level_1_size", "level_2_size", "overlap",
            "output_format", "color",
        ]
        print_error(f"Invalid key: {key}")
        console.print(f"Valid keys: {', '.join(valid_keys)}")
        raise typer.Exit(1)

    # Save to appropriate location
    if user:
        if config.save_user():
            print_success(f"Set {key}={value} in user config")
        else:
            print_error("Failed to save user config (tomli-w not installed?)")
            raise typer.Exit(1)
    elif project:
        if config.save_project():
            print_success(f"Set {key}={value} in project config")
        else:
            print_error("Failed to save project config (tomli-w not installed?)")
            raise typer.Exit(1)
    else:
        # Default to project config
        if config.save_project():
            print_success(f"Set {key}={value} in project config")
        else:
            print_warning("Could not save to file, set via environment:")
            env_key = f"KIDKAZZ_{key.upper()}"
            console.print(f"  export {env_key}={value}")


@app.command("reset")
def reset_config(
    project: bool = typer.Option(
        False,
        "--project",
        "-p",
        help="Reset project config",
    ),
    user: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Reset user config",
    ),
    all_configs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Reset all configs",
    ),
) -> None:
    """Reset configuration to defaults."""
    if not any([project, user, all_configs]):
        print_warning("Specify --project, --global, or --all")
        return

    if project or all_configs:
        path = get_project_config_path()
        if path.exists():
            try:
                path.unlink()
                print_success(f"Removed project config: {path}")
            except Exception as e:
                print_error(f"Failed to remove project config: {e}")
        else:
            console.print(f"Project config not found: {path}")

    if user or all_configs:
        path = get_user_config_path()
        if path.exists():
            try:
                path.unlink()
                print_success(f"Removed user config: {path}")
            except Exception as e:
                print_error(f"Failed to remove user config: {e}")
        else:
            console.print(f"User config not found: {path}")


@app.command("init")
def init_config(
    project: bool = typer.Option(
        True,
        "--project",
        "-p",
        help="Create project config",
    ),
) -> None:
    """Create a default configuration file."""
    defaults = CLIConfig.get_defaults()

    if project:
        path = get_project_config_path()
        if path.exists():
            print_warning(f"Config already exists: {path}")
            return

        if defaults.save_project(path):
            print_success(f"Created project config: {path}")
            console.print("\nEdit the file to customize settings.")
        else:
            print_error("Failed to create config (tomli-w not installed?)")
            console.print("Install with: pip install tomli-w")
            raise typer.Exit(1)
