from pathlib import Path
import yaml


def load_config(config_path):
    """Read a YAML config file."""

    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config



def create_path(path_template, **values):
    """Fill a path template (e.g. from config) with named values."""
    return Path(path_template.format(**values))


def create_timestep_path(path_template, timestep):
    """Fill a path template with a single ``timestep`` value."""
    return create_path(path_template, timestep=timestep)
