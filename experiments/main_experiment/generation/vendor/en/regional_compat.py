"""Single full-frame region support without changing the historical source files.

Historical matrixdealer initializes local variables only when ADDROW/ADDCOL
appears. A legal one-region plan has neither. Construct its one-cell grid
directly using the original Region/Row types; all other layouts call through.
"""
from contextlib import contextmanager
import importlib


def matrix_dispatch(original, region_class, row_class, target):
    def dispatch(state, split_ratio, baseratio):
        if state is target and split_ratio == "1.0":
            base = float(baseratio)
            if not 0 < base <= 1:
                raise ValueError("invalid base ratio")
            state.split_ratio = [row_class(0.0, 1.0, [region_class(0.0, 1.0, base, 0)])]
            state.baseratio = [[base]]
            return None
        return original(state, split_ratio, baseratio)
    return dispatch


@contextmanager
def single_region_compatibility(pipe):
    # The runner holds a process lock and executes one image at a time.
    # Restore the imported module function even when generation raises.
    pipeline_module = importlib.import_module(type(pipe).__module__)
    matrix_module = importlib.import_module("matrix")
    original = pipeline_module.matrixdealer
    pipeline_module.matrixdealer = matrix_dispatch(original, matrix_module.Region, matrix_module.Row, pipe)
    try:
        yield
    finally:
        pipeline_module.matrixdealer = original
