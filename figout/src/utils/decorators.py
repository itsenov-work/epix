class MRODecorator:
    """
    A meta-decorator class for use with MRODecoratedMixin.
    """
    def __init__(self, decorator_func):
        self._wrapper = decorator_func

    def __call__(self, f):
        func = self._wrapper(f)
        func.__setattr__("_wrapper", self._wrapper)
        return func


class MRODecoratedMixin:
    """
    A mixin to keep decorators while subclassing.
    Use as follows:


    class A(MRODecoratedMixin):

        @MRODecorator(some_wrapper)
        def f():
            ...

    class B:
        def f():
            ...

    This will guarantee that B.f() will still be wrapped by some_wrapper.
    """
    def __new__(cls, *args, **kwargs):

        """
        Get all decorated methods from superclasses starting from the class that inherited WrapperObject
        """
        wrapped_methods = {}
        for supercls in cls.__mro__[::-1]:
            if supercls == MRODecoratedMixin or not issubclass(supercls, MRODecoratedMixin):
                continue
            for attr_name in dir(supercls):
                attr = getattr(supercls, attr_name)
                if hasattr(attr, "_wrapper"):
                    wrapped_methods[attr_name] = attr._wrapper

        for attr_name in dir(cls):
            if attr_name not in wrapped_methods:
                continue
            wrapper = wrapped_methods[attr_name]
            attr = getattr(cls, attr_name)
            if not hasattr(attr, "_wrapper"):
                attr = wrapper(attr)

            setattr(cls, attr_name, attr)

        inst = super().__new__(cls)
        return inst
