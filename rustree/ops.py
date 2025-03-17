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

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import rustree._rs as _rs


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    'tree_is_leaf',
]


_T = TypeVar('_T')


def tree_is_leaf(
    tree: _T,
    /,
    is_leaf: Callable[[_T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> bool:
    """Test whether the given object is a leaf node.

    See also :func:`tree_flatten`, :func:`tree_leaves`, and :func:`all_leaves`.

    >>> tree_is_leaf(1)
    True
    >>> tree_is_leaf(None)
    False
    >>> tree_is_leaf(None, none_is_leaf=True)
    True
    >>> tree_is_leaf({'a': 1, 'b': (2, 3)})
    False

    Args:
        tree (pytree): A pytree to check if it is a leaf node.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than a leaf. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A boolean indicating if the given object is a leaf node.
    """
    return _rs.is_leaf(tree, is_leaf, none_is_leaf, namespace)
