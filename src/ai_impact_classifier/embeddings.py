from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=16)
def load_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    model_name: str,
    texts: Iterable[str],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> np.ndarray:
    device = device or default_device()
    model = load_model(model_name, device)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)
