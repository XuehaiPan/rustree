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

from typing import TYPE_CHECKING

import rustree._rs as _rs


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rustree.typing import PyTree, PyTreeSpec, T


__all__ = [
    'MAX_RECURSION_DEPTH',
    'NONE_IS_NODE',
    'NONE_IS_LEAF',
    'tree_flatten',
    'tree_unflatten',
    'tree_leaves',
    'tree_structure',
    'tree_is_leaf',
]


MAX_RECURSION_DEPTH: int = _rs.MAX_RECURSION_DEPTH  # 1000
"""Maximum recursion depth for pytree traversal. It is 1000.

This limit prevents infinite recursion from causing an overflow of the C stack
and crashing Python.
"""
NONE_IS_NODE: bool = False  # literal constant
"""Literal constant that treats :data:`None` as a pytree non-leaf node."""
NONE_IS_LEAF: bool = True  # literal constant
"""Literal constant that treats :data:`None` as a pytree leaf node."""


def tree_flatten(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[list[T], PyTreeSpec]:
    """Flatten a pytree.

    See also :func:`tree_flatten_with_path` and :func:`tree_unflatten`.

    The flattening order (i.e., the order of elements in the output list) is deterministic,
    corresponding to a left-to-right depth-first tree traversal.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    )
    >>> tree_flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [1, 2, 3, 4, None, 5],
        PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    )
    >>> tree_flatten(1)
    ([1], PyTreeSpec(*))
    >>> tree_flatten(None)
    ([], PyTreeSpec(None))
    >>> tree_flatten(None, none_is_leaf=True)
    ([None], PyTreeSpec(*, NoneIsLeaf))

    For unordered dictionaries, :class:`dict` and :class:`collections.defaultdict`, the order is
    dependent on the **sorted** keys in the dictionary. Please use :class:`collections.OrderedDict`
    if you want to keep the keys in the insertion order.

    >>> from collections import OrderedDict
    >>> tree = OrderedDict([('b', (2, [3, 4])), ('a', 1), ('c', None), ('d', 5)])
    >>> tree_flatten(tree)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': None, 'd': *}))
    )
    >>> tree_flatten(tree, none_is_leaf=True)  # doctest: +IGNORE_WHITESPACE
    (
        [2, 3, 4, 1, None, 5],
        PyTreeSpec(OrderedDict({'b': (*, [*, *]), 'a': *, 'c': *, 'd': *}), NoneIsLeaf)
    )

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A pair ``(leaves, treespec)`` where the first element is a list of leaf values and the
        second element is a treespec representing the structure of the pytree.
    """
    return _rs.flatten(tree, is_leaf, none_is_leaf, namespace)


def tree_unflatten(treespec: PyTreeSpec, leaves: Iterable[T]) -> PyTree[T]:
    """Reconstruct a pytree from the treespec and the leaves.

    The inverse of :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> leaves, treespec = tree_flatten(tree)
    >>> tree == tree_unflatten(treespec, leaves)
    True

    Args:
        treespec (PyTreeSpec): The treespec to reconstruct.
        leaves (iterable): The list of leaves to use for reconstruction. The list must match the
            number of leaves of the treespec.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure described by
        ``treespec``.
    """
    return treespec.unflatten(leaves)


def tree_leaves(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> list[T]:
    """Get the leaves of a pytree.

    See also :func:`tree_flatten` and :func:`tree_iter`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_leaves(tree)
    [1, 2, 3, 4, 5]
    >>> tree_leaves(tree, none_is_leaf=True)
    [1, 2, 3, 4, None, 5]
    >>> tree_leaves(1)
    [1]
    >>> tree_leaves(None)
    []
    >>> tree_leaves(None, none_is_leaf=True)
    [None]

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A list of leaf values.
    """
    return _rs.flatten(tree, is_leaf, none_is_leaf, namespace)[0]


def tree_structure(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> PyTreeSpec:
    """Get the treespec for a pytree.

    See also :func:`tree_flatten`.

    >>> tree = {'b': (2, [3, 4]), 'a': 1, 'c': None, 'd': 5}
    >>> tree_structure(tree)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': None, 'd': *})
    >>> tree_structure(tree, none_is_leaf=True)
    PyTreeSpec({'a': *, 'b': (*, [*, *]), 'c': *, 'd': *}, NoneIsLeaf)
    >>> tree_structure(1)
    PyTreeSpec(*)
    >>> tree_structure(None)
    PyTreeSpec(None)
    >>> tree_structure(None, none_is_leaf=True)
    PyTreeSpec(*, NoneIsLeaf)

    Args:
        tree (pytree): A pytree to flatten.
        is_leaf (callable, optional): An optionally specified function that will be called at each
            flattening step. It should return a boolean, with :data:`True` stopping the traversal
            and the whole subtree being treated as a leaf, and :data:`False` indicating the
            flattening should traverse the current object.
        none_is_leaf (bool, optional): Whether to treat :data:`None` as a leaf. If :data:`False`,
            :data:`None` is a non-leaf node with arity 0. Thus :data:`None` is contained in the
            treespec rather than in the leaves list. (default: :data:`False`)
        namespace (str, optional): The registry namespace used for custom pytree node types.
            (default: :const:`''`, i.e., the global namespace)

    Returns:
        A treespec object representing the structure of the pytree.
    """
    return _rs.flatten(tree, is_leaf, none_is_leaf, namespace)[1]


def tree_is_leaf(
    tree: PyTree[T],
    /,
    is_leaf: Callable[[T], bool] | None = None,
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
    return _rs.is_leaf(tree, is_leaf, none_is_leaf, namespace)  # type: ignore[arg-type]
