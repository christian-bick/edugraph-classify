"""Application-boundary configuration helpers."""

from pathlib import Path

from dotenv import load_dotenv


def load_local_environment(env_file: str | Path = ".env") -> bool:
    """Load a local env file while preserving variables supplied by the process."""

    path = Path(env_file)
    if not path.is_file():
        return False
    return load_dotenv(dotenv_path=path, override=False)
