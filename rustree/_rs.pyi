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

# pylint: disable=all

import enum

# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int  # (1UL << 10)

def is_namedtuple(obj: object | type, /) -> bool: ...
def is_namedtuple_instance(obj: object, /) -> bool: ...
def is_namedtuple_class(cls: type, /) -> bool: ...
def namedtuple_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]: ...
def is_structseq(obj: object | type, /) -> bool: ...
def is_structseq_instance(obj: object, /) -> bool: ...
def is_structseq_class(cls: type, /) -> bool: ...
def structseq_fields(obj: tuple | type[tuple], /) -> tuple[str, ...]: ...

class PyTreeKind(enum.IntEnum):
    CUSTOM = 0  # a custom type
    LEAF = enum.auto()  # an opaque leaf node
    NONE = enum.auto()  # None
    TUPLE = enum.auto()  # a tuple
    LIST = enum.auto()  # a list
    DICT = enum.auto()  # a dict
    NAMEDTUPLE = enum.auto()  # a collections.namedtuple
    ORDEREDDICT = enum.auto()  # a collections.OrderedDict
    DEFAULTDICT = enum.auto()  # a collections.defaultdict
    DEQUE = enum.auto()  # a collections.deque
    STRUCTSEQUENCE = enum.auto()  # a PyStructSequence
