from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle, tee
from typing import Any, Callable, Generator, Iterable, Literal, overload

import torch


@overload  # noqa: E302
def iunzip[T, U](
    tups: Iterable[tuple[T, U]], n: Literal[2]
) -> tuple[Iterable[T], Iterable[U]]:
    ...

@overload  # noqa: E302
def iunzip[T, U, V](
    tups: Iterable[tuple[T, U, V]], n: Literal[3]
) -> tuple[Iterable[T], Iterable[U], Iterable[V]]:
    ...

def iunzip(tups: Iterable[tuple], n: int) -> tuple[Iterable, ...]:  # noqa: E302
    # https://stackoverflow.com/a/77797926
    tees = tee(tups, n)

    def select(i: int) -> Generator[Any, None, None]:
        for tup in tees[i]:
            yield tup[i]

    return tuple(select(i) for i in range(n))


def iunsqueeze[T](arg_iter: Iterable[T]) -> Iterable[tuple[T]]:
    for arg in arg_iter:
        yield (arg,)


def imap[*Ts, T](
    inputs: Iterable[tuple[*Ts]],  # noqa: F821
    func: Callable[[*Ts], T],  # noqa: F821
    n_tasks: int | None,
) -> Generator[T, None, None]:
    if n_tasks is None:
        for data_in in inputs:
            yield func(*data_in)
        return

    tasks = deque[Future[T]]()
    with ThreadPoolExecutor(n_tasks) as executor:
        for data_in in inputs:
            # clear out the task queue of completed tasks, then wait until there's room
            while (tasks and tasks[0].done()) or len(tasks) > n_tasks:
                yield tasks.popleft().result()

            task = executor.submit(func, *data_in)
            tasks.append(task)

        # wait for the remaining tasks to finish
        while tasks:
            yield tasks.popleft().result()


def imap_multi_gpu[*Ts, T](
    inputs: Iterable[tuple[*Ts]],  # noqa: F821
    func: Callable[[torch.device, *Ts], T],  # noqa: F821
    n_tasks: int,
) -> Generator[T, None, None]:

    def func_with_gpu(device: torch.device, data_in: tuple[*Ts]) -> T:  # noqa: F821
        data_out = func(device, *data_in)
        return data_out

    devices = cycle(torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()))
    for data_out in imap(zip(devices, inputs), func_with_gpu, n_tasks):
        yield data_out
