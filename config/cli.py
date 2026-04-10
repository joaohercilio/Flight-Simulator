# config/cli.py
"""Command-line interface for the flight simulator."""

from __future__ import annotations

import argparse
import pathlib

from config.settings import SimConfig


def parse_args() -> SimConfig:
    """Parses command-line arguments and loads the simulation config.

    Returns:
        Populated SimConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    parser = argparse.ArgumentParser(
        description="6DOF Flight Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="sim_config.toml",
        type=pathlib.Path,
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    return SimConfig.from_toml_file(args.config)
