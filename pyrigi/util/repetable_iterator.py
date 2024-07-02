from typing import Iterable, Iterator, List


class RepeatableIterator[T](Iterator[T]):
    """
    Wrapper for an iterator that caches all the items returned by the iterator
    in the first pass for future passes. So make sure you exhaust the iterator
    the first time.
    """

    def __init__(self, iterable: Iterable[T]):
        self._iterable = iter(iterable)
        self._is_first = True
        self._cache: List[T] = []

    def __iter__(self) -> Iterator[T]:
        if self._is_first:
            self._is_first = False
            return self
        else:
            return iter(self._cache)

    def __next__(self) -> T:
        item = next(self._iterable)
        self._cache.append(item)
        return item
