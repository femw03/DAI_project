"""A module that contains global helper functions"""

import time
from typing import Callable, List, Tuple, TypeVar

T = TypeVar("T")


def timeit(callable: Callable[[], T]) -> Tuple[T, float]:
    start = time.time()
    result = callable()
    end = time.time()
    return result, end - start


def timeit_n(callable: Callable[[], T], n: int) -> Tuple[List[T], float]:
    """time the callable n times and return each result and the average time"""
    results = []
    average_time = 0.0
    for i in range(n):
        result, time = timeit(callable)
        results.append(result)
        average_time += time / n
    return results, time
