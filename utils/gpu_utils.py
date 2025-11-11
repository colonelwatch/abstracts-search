import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle, tee
from typing import Any, Callable, Concatenate, Generator, Iterable, Literal, overload

import torch


@overload
def iunzip[T, U](
    tups: Iterable[tuple[T, U]], n: Literal[2]
) -> tuple[Iterable[T], Iterable[U]]: ...


@overload
def iunzip[T, U, V](
    tups: Iterable[tuple[T, U, V]], n: Literal[3]
) -> tuple[Iterable[T], Iterable[U], Iterable[V]]: ...


def iunzip(tups: Iterable[tuple], n: int) -> tuple[Iterable, ...]:
    # https://stackoverflow.com/a/77797926
    tees = tee(tups, n)

    def select(i: int) -> Generator[Any, None, None]:
        for tup in tees[i]:
            yield tup[i]

    return tuple(select(i) for i in range(n))


def iunsqueeze[T](arg_iter: Iterable[T]) -> Iterable[tuple[T]]:
    for arg in arg_iter:
        yield (arg,)


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap[T, U_contra](
    inputs: Iterable[tuple[U_contra]],
    func: Callable[[U_contra], T],
    n_tasks: int,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra](
    inputs: Iterable[tuple[U_contra, V_contra]],
    func: Callable[[U_contra, V_contra], T],
    n_tasks: int,
) -> Generator[T, None, None]: ...


@overload
def imap[T, U_contra, V_contra, W_contra](
    inputs: Iterable[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[U_contra, V_contra, W_contra], T],
    n_tasks: int,
) -> Generator[T, None, None]: ...


def imap[T](
    inputs: Iterable[tuple],
    func: Callable[..., T],
    n_tasks: int,
) -> Generator[T, None, None]:
    if n_tasks == 0:
        for data_in in inputs:
            yield func(*data_in)
        return
    elif n_tasks < 0:
        n_tasks = os.cpu_count() or 1

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


# NOTE: didn't use TypeVarTuple because it isn't contravariant
@overload
def imap_multi_gpu[T, U_contra](
    inputs: Iterable[tuple[U_contra]],
    func: Callable[[torch.device, U_contra], T],
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra](
    inputs: Iterable[tuple[U_contra, V_contra]],
    func: Callable[[torch.device, U_contra, V_contra], T],
) -> Generator[T, None, None]: ...


@overload
def imap_multi_gpu[T, U_contra, V_contra, W_contra](
    inputs: Iterable[tuple[U_contra, V_contra, W_contra]],
    func: Callable[[torch.device, U_contra, V_contra, W_contra], T],
) -> Generator[T, None, None]: ...


def imap_multi_gpu[T](
    inputs: Iterable[tuple],
    func: Callable[Concatenate[torch.device, ...], T],
) -> Generator[T, None, None]:
    def func_with_gpu(device: torch.device, data_in: tuple) -> T:
        data_out = func(device, *data_in)
        return data_out

    n_gpus = torch.cuda.device_count()
    devices = cycle(torch.device(f"cuda:{i}") for i in range(n_gpus))
    for data_out in imap(zip(devices, inputs), func_with_gpu, n_gpus):
        yield data_out
