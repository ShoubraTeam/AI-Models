from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator


def configure_startup_noise() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings(
        "ignore",
        message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated since 0\.13.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Falling back to the old format < 1\.6.*",
        category=FutureWarning,
        module=r"torch\.hub",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
        category=UserWarning,
        module=r"multiprocessing\.resource_tracker",
    )

    for logger_name in (
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


@contextlib.contextmanager
def suppress_model_loader_output() -> Iterator[None]:
    configure_startup_noise()

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
