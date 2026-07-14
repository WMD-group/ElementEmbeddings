"""IO utils for AtomicEmbeddings."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, o: Any) -> Any:
        """Encode numpy types."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
