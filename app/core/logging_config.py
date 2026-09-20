import logging
import logging.handlers
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = ROOT / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LOG_FILE = LOG_DIR / "app.log"


def setup_logging():

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            console_handler,
        ],
        force=True,
    )