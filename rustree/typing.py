# Copyright 2024-2025 Xuehai Pan. All Rights Reserved.
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

import abc
import functools
import platform
import sys
import types
from builtins import dict as Dict  # noqa: N812
from builtins import list as List  # noqa: N812
from builtins import tuple as Tuple  # noqa: N812
from collections import OrderedDict
from collections import defaultdict as DefaultDict  # noqa: N812
from collections import deque as Deque  # noqa: N812
from collections.abc import (
    Collection,
    Hashable,
    Iterable,
    Sequence,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Optional,
    Protocol,
    TypeVar,
    final,
    runtime_checkable,
)
from typing_extensions import (
    NamedTuple,  # Generic NamedTuple: Python 3.11+
    Never,  # Python 3.11+
    ParamSpec,  # Python 3.10+
    Self,  # Python 3.11+
    TypeAlias,  # Python 3.10+
)

import rustree._rs as _rs
from rustree._rs import PyTreeKind
from rustree.accessors import (
    AutoEntry,
    DataclassEntry,
    FlattenedEntry,
    GetAttrEntry,
    GetItemEntry,
    MappingEntry,
    NamedTupleEntry,
    PyTreeAccessor,
    PyTreeEntry,
    SequenceEntry,
    StructSequenceEntry,
)


__all__ = [
    'PyTreeKind',
    'Children',
    'MetaData',
    'FlattenFunc',
    'UnflattenFunc',
    'PyTreeEntry',
    'GetItemEntry',
    'GetAttrEntry',
    'FlattenedEntry',
    'AutoEntry',
    'SequenceEntry',
    'MappingEntry',
    'NamedTupleEntry',
    'StructSequenceEntry',
    'DataclassEntry',
    'PyTreeAccessor',
    'is_namedtuple',
    'is_namedtuple_class',
    'is_namedtuple_instance',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_class',
    'is_structseq_instance',
    'structseq_fields',
    'T',
    'S',
    'U',
    'KT',
    'VT',
    'P',
    'F',
    'Iterable',
    'Sequence',
    'Tuple',
    'List',
    'Dict',
    'NamedTuple',
    'OrderedDict',
    'DefaultDict',
    'Deque',
    'StructSequence',
]


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
KT = TypeVar('KT')
VT = TypeVar('VT')
P = ParamSpec('P')
F = TypeVar('F', bound=Callable[..., Any])


Children: TypeAlias = Iterable[T]
MetaData: TypeAlias = Optional[Hashable]


@runtime_checkable
class CustomTreeNode(Protocol[T]):
    """The abstract base class for custom pytree nodes."""

    def tree_flatten(
        self,
        /,
    ) -> (
        # Use `range(num_children)` as path entries
        tuple[Children[T], MetaData]
        |
        # With optionally implemented path entries
        tuple[Children[T], MetaData, Iterable[Any] | None]
    ):
        """Flatten the custom pytree node into children and metadata."""

    @classmethod
    def tree_unflatten(cls, metadata: MetaData, children: Children[T], /) -> Self:
        """Unflatten the children and metadata into the custom pytree node."""


class FlattenFunc(Protocol[T]):  # pylint: disable=too-few-public-methods
    """The type stub class for flatten functions."""

    @abc.abstractmethod
    def __call__(
        self,
        container: Collection[T],
        /,
    ) -> tuple[Children[T], MetaData] | tuple[Children[T], MetaData, Iterable[Any] | None]:
        """Flatten the container into children and metadata."""


class UnflattenFunc(Protocol[T]):  # pylint: disable=too-few-public-methods
    """The type stub class for unflatten functions."""

    @abc.abstractmethod
    def __call__(self, metadata: MetaData, children: Children[T], /) -> Collection[T]:
        """Unflatten the children and metadata back into the container."""


def _override_with_(
    rust_implementation: Callable[P, T],
    /,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to override the Python implementation with the C++ implementation.

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
        # pylint: disable-next=unidiomatic-typecheck
        and all(type(field) is str for field in cls._fields)  # type: ignore[attr-defined]
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
        >>> issubclass(time.struct_time, StructSequence)
        True
        >>> class MyTuple(tuple):
        ...     n_fields = 2
        ...     n_sequence_fields = 2
        ...     n_unnamed_fields = 0
        >>> issubclass(MyTuple, StructSequence)
        False
        """
        return is_structseq_class(subclass)

    def __instancecheck__(cls, instance: Any, /) -> bool:
        """Return whether the object is a PyStructSequence instance.

        >>> import sys
        >>> isinstance(sys.float_info, StructSequence)
        True
        >>> isinstance((1, 2), StructSequence)
        False
        """
        return is_structseq_instance(instance)


# Reference: https://github.com/python/typeshed/blob/main/stdlib/_typeshed/__init__.pyi
# This is an internal CPython type that is like, but subtly different from a NamedTuple.
# `StructSequence` classes are unsubclassable, so are all decorated with `@final`.
# pylint: disable-next=invalid-name,missing-class-docstring
@final
class StructSequence(tuple[_T_co, ...], metaclass=StructSequenceMeta):
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    __slots__: ClassVar[tuple[()]] = ()

    n_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name
    n_sequence_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name
    n_unnamed_fields: Final[ClassVar[int]]  # type: ignore[misc] # pylint: disable=invalid-name

    def __init_subclass__(cls, /) -> Never:
        """Prohibit subclassing."""
        raise TypeError("type 'StructSequence' is not an acceptable base type")

    # pylint: disable-next=unused-argument,redefined-builtin
    def __new__(cls, /, sequence: Iterable[_T_co], dict: dict[str, Any] = ...) -> Self:
        """Create a new :class:`StructSequence` instance."""
        raise NotImplementedError


structseq: TypeAlias = StructSequence  # noqa: PYI042

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


# pylint: disable-next=line-too-long
StructSequenceFieldType: type[types.MemberDescriptorType] = type(type(sys.version_info).major)  # type: ignore[assignment]


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
        indices_by_name = {
            name: member.index  # type: ignore[attr-defined]
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        }
        fields = sorted(indices_by_name, key=indices_by_name.get)  # type: ignore[arg-type]
    else:
        fields = [
            name
            for name, member in vars(cls).items()
            if isinstance(member, StructSequenceFieldType)
        ]
    return tuple(fields[: cls.n_sequence_fields])  # type: ignore[attr-defined]
