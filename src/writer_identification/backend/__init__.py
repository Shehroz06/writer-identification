"""Writer-identification adapter code: composes the shared engine
(preprocessing + embeddings) into an `.identify(image)` gallery-ranking call,
loaded with this child project's own trained checkpoint (see `config.py`)."""

from writer_identification.backend.adapter import (
    IdentificationAdapter,
    IdentificationMatch,
)
from writer_identification.backend.config import IdentificationAppConfig

__all__ = [
    "IdentificationAdapter",
    "IdentificationAppConfig",
    "IdentificationMatch",
]
