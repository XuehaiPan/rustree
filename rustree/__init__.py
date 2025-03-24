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
"""RusTree: Optimized PyTree Utilities written in Rust."""

from rustree import accessors, typing
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
from rustree.ops import (
    MAX_RECURSION_DEPTH,
    NONE_IS_LEAF,
    NONE_IS_NODE,
    tree_flatten,
    tree_is_leaf,
    tree_leaves,
    tree_structure,
)
from rustree.typing import (
    CustomTreeNode,
    FlattenFunc,
    PyTree,
    PyTreeKind,
    PyTreeSpec,
    PyTreeTypeVar,
    UnflattenFunc,
    is_namedtuple,
    is_namedtuple_class,
    is_namedtuple_instance,
    is_structseq,
    is_structseq_class,
    is_structseq_instance,
    namedtuple_fields,
    structseq_fields,
)


__all__ = [
    # Tree operations
    'MAX_RECURSION_DEPTH',
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_leaves',
    'tree_structure',
    'tree_is_leaf',
    # Typing
    'PyTreeSpec',
    'PyTreeKind',
    'PyTree',
    'PyTreeTypeVar',
    'CustomTreeNode',
    'FlattenFunc',
    'UnflattenFunc',
    'is_namedtuple',
    'is_namedtuple_class',
    'is_namedtuple_instance',
    'namedtuple_fields',
    'is_structseq',
    'is_structseq_instance',
    'is_structseq_class',
    'structseq_fields',
    # Accessor
    'PyTreeEntry',
    'GetAttrEntry',
    'GetItemEntry',
    'FlattenedEntry',
    'AutoEntry',
    'SequenceEntry',
    'MappingEntry',
    'NamedTupleEntry',
    'StructSequenceEntry',
    'DataclassEntry',
    'PyTreeAccessor',
]
