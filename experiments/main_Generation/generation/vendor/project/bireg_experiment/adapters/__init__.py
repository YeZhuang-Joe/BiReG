"""Image-generation method adapters.

Heavy GPU dependencies are imported lazily inside each adapter so manifest
validation and dry runs remain usable on CPU-only machines.
"""

from .base import AdapterError, GenerationAdapter
from .kolors import KolorsAdapter
from .rpg import RPGAdapter
from .rpg_kolors import RPGKolorsAdapter
from .sdxl import SDXLAdapter

__all__ = [
    "AdapterError", "GenerationAdapter", "KolorsAdapter", "RPGAdapter",
    "RPGKolorsAdapter", "SDXLAdapter"
]
