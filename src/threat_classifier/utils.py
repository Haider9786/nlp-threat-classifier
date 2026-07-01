"""Shared utilities: config loading and logging setup."""
import logging
import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config file.

    Args:
        path: Path to the config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, config: dict) -> logging.Logger:
    """Create a configured logger that writes to console and file.

    Args:
        name: Logger name (usually __name__ of the calling module).
        config: Loaded config dict, expects config['logging'].

    Returns:
        Configured Logger instance.
    """
    log_config = config.get("logging", {})
    log_file = log_config.get("log_file", "logs/app.log")
    level = getattr(logging, log_config.get("level", "INFO"))

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger