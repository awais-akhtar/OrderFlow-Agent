"""Pluggable text and image feature backends for menu items."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageStat


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MenuRepresentation:
    text: np.ndarray
    image: np.ndarray
    has_image: bool

    @property
    def joint(self) -> np.ndarray:
        return _normalise(np.concatenate([self.text, self.image]))


class MenuEmbeddingBackend(Protocol):
    name: str

    def represent(self, text: str, image_bytes: bytes | None = None) -> MenuRepresentation: ...


class LightweightMenuEncoder:
    """Dependency-light hashed text plus transparent colour/composition features."""

    name = "local-hashed-text-plus-image-features"

    def __init__(self, text_dimensions: int = 192) -> None:
        if text_dimensions < 32:
            raise ValueError("text_dimensions must be at least 32.")
        self.text_dimensions = text_dimensions

    def represent(self, text: str, image_bytes: bytes | None = None) -> MenuRepresentation:
        text_vector = np.zeros(self.text_dimensions, dtype=np.float32)
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.text_dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            text_vector[index] += sign
        text_vector = _normalise(text_vector)
        image_vector = local_image_features(image_bytes)
        return MenuRepresentation(text_vector, image_vector, image_bytes is not None)


class ProviderMenuEncoder(LightweightMenuEncoder):
    """Provider text embeddings with the same local, inspectable image features."""

    def __init__(self, provider) -> None:
        self.provider = provider
        self.name = f"{getattr(provider, 'provider_id', 'provider')}-text-plus-local-image"

    def represent(self, text: str, image_bytes: bytes | None = None) -> MenuRepresentation:
        vectors = self.provider.embed_texts([text])
        if len(vectors) != 1:
            raise ValueError("Embedding provider returned an unexpected number of vectors.")
        text_vector = _normalise(np.asarray(vectors[0], dtype=np.float32))
        return MenuRepresentation(text_vector, local_image_features(image_bytes), image_bytes is not None)


def local_image_features(image_bytes: bytes | None) -> np.ndarray:
    """Return 20 fixed features; all zeros explicitly represents a missing image."""
    if not image_bytes:
        return np.zeros(20, dtype=np.float32)
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < 2 or height < 2:
            raise ValueError("Menu images must be at least 2 by 2 pixels.")
        statistic = ImageStat.Stat(rgb)
        means = np.asarray(statistic.mean, dtype=np.float32) / 255.0
        deviations = np.asarray(statistic.stddev, dtype=np.float32) / 255.0
        thumbnail = np.asarray(rgb.resize((2, 2)), dtype=np.float32).reshape(-1) / 255.0
        geometry = np.asarray([min(width / height, 4.0) / 4.0, min(height / width, 4.0) / 4.0], dtype=np.float32)
        return _normalise(np.concatenate([means, deviations, thumbnail, geometry]))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or not right.size or left.shape != right.shape:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
