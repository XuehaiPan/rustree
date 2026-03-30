# Copyright 2022-2026 MetaOPT Team. All Rights Reserved.
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

# pylint: disable=missing-function-docstring,invalid-name

import dataclasses
import itertools
import re
from collections import OrderedDict, UserDict, UserList, defaultdict, deque
from typing import Any, NamedTuple

import pytest

import rustree
from helpers import TREE_ACCESSORS, SysFloatInfoType, assert_equal_type_and_value, parametrize


def test_pytree_accessor_new():
    assert_equal_type_and_value(rustree.PyTreeAccessor(), rustree.PyTreeAccessor(()))
    assert_equal_type_and_value(
        rustree.PyTreeAccessor(
            [
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ],
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        rustree.PyTreeAccessor(
            [
                rustree.MappingEntry('a', dict, rustree.PyTreeKind.DICT),
                rustree.MappingEntry('b', dict, rustree.PyTreeKind.DICT),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ],
        ),
        rustree.PyTreeAccessor(
            (rustree.MappingEntry(key, dict, rustree.PyTreeKind.DICT) for key in ('a', 'b', 'c')),
        ),
    )

    with pytest.raises(TypeError, match=r'Expected a path of PyTreeEntry, got .*\.'):
        rustree.PyTreeAccessor([rustree.MappingEntry('a', dict, rustree.PyTreeKind.DICT), 'b'])


def test_pytree_accessor_add():
    assert_equal_type_and_value(
        rustree.PyTreeAccessor() + rustree.PyTreeAccessor(),
        rustree.PyTreeAccessor(),
    )
    assert_equal_type_and_value(
        rustree.PyTreeAccessor() + rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
        rustree.PyTreeAccessor((rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),)),
    )
    assert_equal_type_and_value(
        rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE) + rustree.PyTreeAccessor(),
        rustree.PyTreeAccessor((rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),)),
    )
    assert_equal_type_and_value(
        (
            rustree.PyTreeAccessor()
            + rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE)
            + rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST)
            + rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT)
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE)
            + rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST)
            + rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT)
            + rustree.PyTreeAccessor()
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE)
            + rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST)
            + rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT)
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        (
            rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE)
            + rustree.PyTreeAccessor(
                (
                    rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                    rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
                ),
            )
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )

    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        rustree.PyTreeAccessor() + 'a'
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        1 + rustree.PyTreeAccessor()
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE) + 'a'
    with pytest.raises(TypeError, match=re.escape(r'unsupported operand type(s) for +')):
        1 + rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE)


def test_pytree_accessor_mul():
    assert_equal_type_and_value(rustree.PyTreeAccessor() * 3, rustree.PyTreeAccessor())
    assert 3 * rustree.PyTreeAccessor() == rustree.PyTreeAccessor()
    assert_equal_type_and_value(
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        )
        * 2,
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )
    assert_equal_type_and_value(
        2
        * rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
        rustree.PyTreeAccessor(
            (
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
                rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
                rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
                rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
            ),
        ),
    )


def test_pytree_accessor_getitem():
    entries = (
        rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.TUPLE),
        rustree.SequenceEntry(1, list, rustree.PyTreeKind.LIST),
        rustree.MappingEntry('c', dict, rustree.PyTreeKind.DICT),
    )
    accessor = rustree.PyTreeAccessor(entries)

    for i in range(-len(entries) - 2, len(entries) + 1):
        if -len(entries) <= i < len(entries):
            assert_equal_type_and_value(accessor[i], entries[i])
        else:
            with pytest.raises(IndexError, match=r'index out of range'):
                accessor[i]

        for j in range(-len(entries) - 2, len(entries) + 1):
            assert len(accessor[i:j]) == len(entries[i:j])
            assert_equal_type_and_value(accessor[i:j], rustree.PyTreeAccessor(entries[i:j]))


@parametrize(
    none_is_leaf=[False, True],
)
def test_pytree_accessor_equal_hash(none_is_leaf):
    for i, accessor1 in enumerate(itertools.chain.from_iterable(TREE_ACCESSORS[none_is_leaf])):
        for j, accessor2 in enumerate(itertools.chain.from_iterable(TREE_ACCESSORS[none_is_leaf])):
            if i == j:
                assert accessor1 == accessor2
                assert hash(accessor1) == hash(accessor2)
            if accessor1 == accessor2:
                assert hash(accessor1) == hash(accessor2)
            else:
                assert hash(accessor1) != hash(accessor2)


def test_pytree_entry_init():
    for path_entry_type in (
        rustree.PyTreeEntry,
        rustree.GetAttrEntry,
        rustree.GetItemEntry,
        rustree.FlattenedEntry,
        rustree.AutoEntry,
        rustree.SequenceEntry,
        rustree.MappingEntry,
        rustree.NamedTupleEntry,
        rustree.StructSequenceEntry,
        rustree.DataclassEntry,
    ):
        entry = path_entry_type(0, int, rustree.PyTreeKind.CUSTOM)
        assert entry.entry == 0
        assert entry.type is int
        assert entry.kind == rustree.PyTreeKind.CUSTOM

        with pytest.raises(
            ValueError,
            match=(
                re.escape('Cannot create a leaf path entry.')
                if path_entry_type is not rustree.AutoEntry
                else r'Cannot create an automatic path entry for PyTreeKind .*\.'
            ),
        ):
            path_entry_type(0, int, rustree.PyTreeKind.LEAF)
        with pytest.raises(
            ValueError,
            match=(
                re.escape('Cannot create a path entry for None.')
                if path_entry_type is not rustree.AutoEntry
                else r'Cannot create an automatic path entry for PyTreeKind .*\.'
            ),
        ):
            path_entry_type(None, type(None), rustree.PyTreeKind.NONE)


def test_auto_entry_new_invalid_kind():
    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        rustree.AutoEntry(0, int, rustree.PyTreeKind.LEAF)

    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        rustree.AutoEntry(None, type(None), rustree.PyTreeKind.NONE)

    with pytest.raises(
        ValueError,
        match=r'Cannot create an automatic path entry for PyTreeKind .*\.',
    ):
        rustree.AutoEntry(0, tuple, rustree.PyTreeKind.TUPLE)

    class SubclassedAutoEntry(rustree.AutoEntry):
        pass

    with pytest.raises(ValueError, match=re.escape('Cannot create a leaf path entry.')):
        SubclassedAutoEntry(0, int, rustree.PyTreeKind.LEAF)

    with pytest.raises(ValueError, match=re.escape('Cannot create a path entry for None.')):
        SubclassedAutoEntry(None, type(None), rustree.PyTreeKind.NONE)

    assert_equal_type_and_value(
        SubclassedAutoEntry(0, tuple, rustree.PyTreeKind.TUPLE),
        rustree.PyTreeEntry(0, tuple, rustree.PyTreeKind.TUPLE),
        expected_type=SubclassedAutoEntry,
    )


def test_auto_entry_new_dispatch():
    class CustomTuple(NamedTuple):
        x: Any
        y: Any
        z: Any

    @dataclasses.dataclass
    class CustomDataclass:
        foo: Any
        bar: Any

    class MyMapping(UserDict):
        pass

    class MySequence(UserList):
        pass

    class MyObject:
        pass

    assert_equal_type_and_value(
        rustree.AutoEntry(0, SysFloatInfoType, rustree.PyTreeKind.CUSTOM),
        rustree.StructSequenceEntry(0, SysFloatInfoType, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.StructSequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, CustomTuple, rustree.PyTreeKind.CUSTOM),
        rustree.NamedTupleEntry(0, CustomTuple, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.NamedTupleEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry('foo', CustomDataclass, rustree.PyTreeKind.CUSTOM),
        rustree.DataclassEntry('foo', CustomDataclass, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.DataclassEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry('foo', dict, rustree.PyTreeKind.CUSTOM),
        rustree.MappingEntry('foo', dict, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.MappingEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry('foo', OrderedDict, rustree.PyTreeKind.CUSTOM),
        rustree.MappingEntry('foo', OrderedDict, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.MappingEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry('foo', defaultdict, rustree.PyTreeKind.CUSTOM),
        rustree.MappingEntry('foo', defaultdict, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.MappingEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry('foo', MyMapping, rustree.PyTreeKind.CUSTOM),
        rustree.MappingEntry('foo', MyMapping, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.MappingEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, tuple, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, tuple, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, list, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, list, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, deque, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, deque, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, str, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, str, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, bytes, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, bytes, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, MySequence, rustree.PyTreeKind.CUSTOM),
        rustree.SequenceEntry(0, MySequence, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.SequenceEntry,
    )

    assert_equal_type_and_value(
        rustree.AutoEntry(0, MyObject, rustree.PyTreeKind.CUSTOM),
        rustree.FlattenedEntry(0, MyObject, rustree.PyTreeKind.CUSTOM),
        expected_type=rustree.FlattenedEntry,
    )

    class SubclassedAutoEntry(rustree.AutoEntry):
        pass

    assert_equal_type_and_value(
        SubclassedAutoEntry(0, MyObject, rustree.PyTreeKind.CUSTOM),
        rustree.PyTreeEntry(0, MyObject, rustree.PyTreeKind.CUSTOM),
        expected_type=SubclassedAutoEntry,
    )


def test_flattened_entry_call():
    @rustree.register_pytree_node_class(namespace='namespace')
    class MyObject:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __eq__(self, other):
            return isinstance(other, MyObject) and (self.x, self.y, self.z) == (
                other.x,
                other.y,
                other.z,
            )

        def __hash__(self):
            return hash((self.x, self.y, self.z))

        def __tree_flatten__(self):
            return (self.x, self.y, self.z), None

        @classmethod
        def __tree_unflatten__(cls, metadata, children):
            return cls(*children)

    obj = MyObject(1, 2, 3)
    expected_accessors = [
        rustree.PyTreeAccessor(
            (rustree.FlattenedEntry(0, MyObject, rustree.PyTreeKind.CUSTOM),),
        ),
        rustree.PyTreeAccessor(
            (rustree.FlattenedEntry(1, MyObject, rustree.PyTreeKind.CUSTOM),),
        ),
        rustree.PyTreeAccessor(
            (rustree.FlattenedEntry(2, MyObject, rustree.PyTreeKind.CUSTOM),),
        ),
    ]

    accessors, leaves, _ = rustree.tree_flatten_with_accessor(obj, namespace='namespace')
    assert leaves == [1, 2, 3]
    assert accessors == expected_accessors
    for a, b in zip(accessors, expected_accessors):
        assert_equal_type_and_value(a, b)

    for accessor in accessors:
        with pytest.raises(TypeError, match=r"<class '.*'> cannot access through .* via entry .*"):
            accessor(obj)
