"""Upscale, clean up and colour-grade images and videos on the GPU."""
from .look import PRESETS, RESOLUTIONS, Look, Processor, cinema_grade, finish, make_look, parse_size
from .model import DEFAULT_MODEL, MODELS, fetch_weights, load_model

__version__ = "0.1.0"
__all__ = ["PRESETS", "RESOLUTIONS", "Look", "Processor", "cinema_grade", "finish", "make_look", "parse_size",
           "DEFAULT_MODEL", "MODELS", "fetch_weights", "load_model", "__version__"]
