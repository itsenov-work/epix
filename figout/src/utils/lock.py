import functools
import threading


def synchronized(wrapped=None, lock=None):
    if wrapped is None:
        return functools.partial(synchronized, lock=lock)
    if lock is None:
        lock = threading.RLock()

    @functools.wraps(wrapped)
    def _wrapper(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)

    return _wrapper


def synchronized_object(wrapped=None, lock_name=None):
    if wrapped is None:
        return functools.partial(synchronized_object, lock_name=lock_name)

    @functools.wraps(wrapped)
    def _wrapper(self, *args, **kwargs):

        with getattr(self, lock_name):
            return wrapped(self, *args, **kwargs)

    return _wrapper
