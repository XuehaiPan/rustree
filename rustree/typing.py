# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Typing utilities for RusTree."""

from __future__ import annotations

import functools
import platform
import types
from typing import Any, Callable, Final, Iterable, Tuple, TypeVar
from typing_extensions import (
    Never,  # Python 3.11+
    ParamSpec,  # Python 3.10+
    Self,  # Python 3.11+
)

import rustree._rs as _rs
from rustree._rs import PyTreeKind


__all__ = [
    'PyTreeKind',
    'is_namedtuple',
    'is_namedtuple_instance',
    'is_namedtuple_class',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_instance',
    'is_structseq_class',
    'structseq_fields',
    'T',
    'S',
    'U',
    'KT',
    'VT',
    'P',
    'F',
]


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')
P = ParamSpec('P')
F = TypeVar('F', bound=Callable[..., Any])


def _override_with_(
    rust_implementation: Callable[P, T],
    /,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to override the Python implementation with the Rust implementation.

    >>> @_override_with_(any)
    ... def my_any(iterable):
    ...     for elem in iterable:
    ...         if elem:
    ...             return True
    ...     return False
    ...
    >>> my_any([False, False, True, False, False, True])  # run at C speed
    True
    """

    def wrapper(python_implementation: Callable[P, T], /) -> Callable[P, T]:
        @functools.wraps(python_implementation)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            return rust_implementation(*args, **kwargs)

        wrapped.__rust_implementation__ = rust_implementation  # type: ignore[attr-defined]
        wrapped.__python_implementation__ = python_implementation  # type: ignore[attr-defined]

        return wrapped

    return wrapper


@_override_with_(_rs.is_namedtuple)
def is_namedtuple(obj: object | type, /) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


@_override_with_(_rs.is_namedtuple_instance)
def is_namedtuple_instance(obj: object, /) -> bool:
    """Return whether the object is an instance of namedtuple."""
    return is_namedtuple_class(type(obj))


@_override_with_(_rs.is_namedtuple_class)
def is_namedtuple_class(cls: type, /) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, '_fields', None), tuple)
        and all(
            type(field) is str  # pylint: disable=unidiomatic-typecheck
            for field in cls._fields  # type: ignore[attr-defined]
        )
        and callable(getattr(cls, '_make', None))
        and callable(getattr(cls, '_asdict', None))
    )


@_override_with_(_rs.namedtuple_fields)
def namedtuple_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]:
    """Return the field names of a namedtuple."""
    if isinstance(obj, type):
        cls = obj
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected a collections.namedtuple type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_namedtuple_class(cls):
            raise TypeError(f'Expected an instance of collections.namedtuple type, got {obj!r}.')
    return cls._fields  # type: ignore[attr-defined]


_T_co = TypeVar('_T_co', covariant=True)


class StructSequenceMeta(type):
    """The metaclass for PyStructSequence stub type."""

    def __subclasscheck__(cls, subclass: type, /) -> bool:
        """Return whether the class is a PyStructSequence type.

        >>> import time
        >>> issubclass(time.struct_time, structseq)
        True
        >>> class MyTuple(tuple):
        ...     n_fields = 2
        ...     n_sequence_fields = 2
        ...     n_unnamed_fields = 0
        >>> issubclass(MyTuple, structseq)
        False
        """
        return is_structseq_class(subclass)

    def __instancecheck__(cls, instance: Any, /) -> bool:
        """Return whether the object is a PyStructSequence instance.

        >>> import sys
        >>> isinstance(sys.float_info, structseq)
        True
        >>> isinstance((1, 2), structseq)
        False
        """
        return is_structseq_instance(instance)


# Reference: https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
# This is an internal CPython type that is like, but subtly different from a NamedTuple.
# `structseq` classes are unsubclassable, so are all decorated with `@final`.
# pylint: disable-next=invalid-name,missing-class-docstring
class structseq(Tuple[_T_co, ...], metaclass=StructSequenceMeta):  # type: ignore[misc] # noqa: N801
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    n_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name
    n_sequence_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name
    n_unnamed_fields: Final[int]  # type: ignore[misc] # pylint: disable=invalid-name

    def __init_subclass__(cls, /) -> Never:
        """Prohibit subclassing."""
        raise TypeError("type 'structseq' is not an acceptable base type")

    # pylint: disable-next=unused-argument,redefined-builtin
    def __new__(cls, /, sequence: Iterable[_T_co], dict: dict[str, Any] = ...) -> Self:
        raise NotImplementedError


del StructSequenceMeta


@_override_with_(_rs.is_structseq)
def is_structseq(obj: object | type, /) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


@_override_with_(_rs.is_structseq_instance)
def is_structseq_instance(obj: object, /) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int = _rs.Py_TPFLAGS_BASETYPE  # (1UL << 10)


@_override_with_(_rs.is_structseq_class)
def is_structseq_class(cls: type, /) -> bool:
    """Return whether the class is a class of PyStructSequence."""
    if (
        isinstance(cls, type)
        # Check direct inheritance from `tuple` rather than `issubclass(cls, tuple)`
        and cls.__bases__ == (tuple,)
        # Check PyStructSequence members
        and isinstance(getattr(cls, 'n_fields', None), int)
        and isinstance(getattr(cls, 'n_sequence_fields', None), int)
        and isinstance(getattr(cls, 'n_unnamed_fields', None), int)
    ):
        # Check the type does not allow subclassing
        if platform.python_implementation() == 'PyPy':
            try:
                types.new_class('subclass', bases=(cls,))
            except (AssertionError, TypeError):
                return True
            return False
        return not bool(cls.__flags__ & Py_TPFLAGS_BASETYPE)
    return False


@_override_with_(_rs.structseq_fields)
def structseq_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]:
    """Return the field names of a PyStructSequence."""
    if isinstance(obj, type):
        cls = obj
        if not is_structseq_class(cls):
            raise TypeError(f'Expected a PyStructSequence type, got {cls!r}.')
    else:
        cls = type(obj)
        if not is_structseq_class(cls):
            raise TypeError(f'Expected an instance of PyStructSequence type, got {obj!r}.')

    if platform.python_implementation() == 'PyPy':
        # pylint: disable-next=import-error,import-outside-toplevel
        from _structseq import structseqfield

        indices_by_name = {
            name: member.index
            for name, member in vars(cls).items()
            if isinstance(member, structseqfield)
        }
        fields = sorted(indices_by_name, key=indices_by_name.get)  # type: ignore[arg-type]
    else:
        fields = [
            name
            for name, member in vars(cls).items()
            if isinstance(member, types.MemberDescriptorType)
        ]
    return tuple(fields[: cls.n_sequence_fields])  # type: ignore[attr-defined]
