"""A module that contains global helper functions"""

import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


def timeit(callable: Callable[[], T]) -> Tuple[T, float]:
    start = time.time()
    result = callable()
    end = time.time()
    return result, end - start
