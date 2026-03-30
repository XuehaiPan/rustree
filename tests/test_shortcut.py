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

import importlib
import operator
import re
import sys
import types
from collections import UserList

import pytest

import rustree
from helpers import GLOBAL_NAMESPACE


def test_pytree_reexports():
    tree_operations = [name[len('tree_') :] for name in rustree.__all__ if name.startswith('tree_')]
    assert rustree.pytree.__all__ == [
        'reexport',
        'PyTreeSpec',
        'PyTreeKind',
        'PyTreeEntry',
        *tree_operations,
        'register_node',
        'register_node_class',
        'unregister_node',
        'dict_insertion_ordered',
    ]

    assert rustree.pytree.PyTreeSpec is rustree.PyTreeSpec
    assert rustree.pytree.PyTreeKind is rustree.PyTreeKind
    assert rustree.pytree.PyTreeEntry is rustree.PyTreeEntry
    for name in tree_operations:
        assert getattr(rustree.pytree, name) is getattr(rustree, f'tree_{name}')
    assert rustree.pytree.register_node is rustree.register_pytree_node
    assert rustree.pytree.register_node_class is rustree.register_pytree_node_class
    assert rustree.pytree.unregister_node is rustree.unregister_pytree_node
    assert rustree.pytree.dict_insertion_ordered is rustree.dict_insertion_ordered
    assert rustree.pytree.__version__ == rustree.__version__


def test_treespec_reexports():
    treespec_operations = [
        name[len('treespec_') :] for name in rustree.__all__ if name.startswith('treespec_')
    ]
    treespec_all_set = set(rustree.treespec.__all__)
    assert treespec_all_set.issubset(treespec_operations)
    assert rustree.treespec.__all__ == [
        name for name in treespec_operations if name in treespec_all_set
    ]

    for name in rustree.treespec.__all__:
        assert getattr(rustree.treespec, name) is getattr(rustree, f'treespec_{name}')


def test_pytree_reexport_with_invalid_argument():
    with pytest.raises(TypeError, match=r'The namespace must be a string'):
        rustree.pytree.reexport(namespace=123)
    rustree.pytree.reexport(namespace='', module='some_module1')
    rustree.pytree.reexport(namespace=GLOBAL_NAMESPACE, module='some_module2')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module='')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module='123abc')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module='abc.123def')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module='abc-def')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module=' ')
    with pytest.raises(ValueError, match=r'invalid module name'):
        rustree.pytree.reexport(namespace='some-namespace', module=' abc')


def check_reexported_module(*, reexported, module, namespace):
    assert reexported.__name__ == module
    assert reexported.dataclasses.__name__ == f'{module}.dataclasses'
    assert reexported.functools.__name__ == f'{module}.functools'
    for mod in (reexported, reexported.dataclasses, reexported.functools):
        assert mod.__name__ in sys.modules
        assert mod is sys.modules[mod.__name__]
        assert importlib.import_module(mod.__name__) is mod
        assert type(mod) is rustree.pytree.ReexportedModule

    assert 'dataclasses' not in reexported.__all__
    assert 'functools' not in reexported.__all__
    assert '__version__' not in reexported.__all__
    assert 'reexport' not in reexported.__all__
    assert 'dataclasses' in dir(reexported)
    assert 'functools' in dir(reexported)
    assert '__version__' in dir(reexported)
    assert 'reexport' not in dir(reexported)
    assert reexported.__version__ == rustree.__version__
    assert reexported.PyTreeSpec is rustree.PyTreeSpec
    assert reexported.PyTreeKind is rustree.PyTreeKind
    assert reexported.PyTreeEntry is rustree.PyTreeEntry

    with pytest.raises(AttributeError, match=r'has no attribute'):
        _ = reexported.not_exist

    for mod in (reexported, reexported.dataclasses, reexported.functools):
        assert set(mod.__all__).issubset(dir(mod))
        for name in dir(mod):
            _ = getattr(mod, name)

    assert reexported.functools.partial is rustree.functools.partial

    @reexported.dataclasses.dataclass
    class MyDataClass:
        x: int
        y: int
        z: int

    class MyList(UserList):
        pass

    reexported.register_node(
        MyList,
        lambda x: (reversed(x), None),
        lambda _, x: MyList(reversed(x)),
    )

    def check_roundtrip(tree):
        leaves, treespec = rustree.tree_flatten(tree)
        assert rustree.tree_unflatten(treespec, leaves) == tree
        leaves, treespec = rustree.tree_flatten(tree, namespace=namespace)
        assert rustree.tree_unflatten(treespec, leaves) == tree
        leaves, treespec = reexported.flatten(tree)
        assert reexported.unflatten(treespec, leaves) == tree
        leaves, treespec = reexported.flatten(tree, namespace='')
        assert reexported.unflatten(treespec, leaves) == tree

    assert rustree.tree_leaves(MyDataClass(1, 2, 3)) == [MyDataClass(1, 2, 3)]
    assert rustree.tree_leaves(MyDataClass(1, 2, 3), namespace=namespace) == [1, 2, 3]
    assert reexported.leaves(MyDataClass(1, 2, 3)) == [1, 2, 3]
    assert reexported.leaves(MyDataClass(1, 2, 3), namespace='') == [MyDataClass(1, 2, 3)]
    check_roundtrip(MyDataClass(1, 2, 3))
    assert reexported.functools.reduce(operator.add, MyDataClass(1, 2, 3)) == 6

    assert rustree.tree_leaves(MyList([1, 2, 3, 4])) == [MyList([1, 2, 3, 4])]
    assert rustree.tree_leaves(MyList([1, 2, 3, 4]), namespace=namespace) == [4, 3, 2, 1]
    assert reexported.leaves(MyList([1, 2, 3, 4])) == [4, 3, 2, 1]
    assert reexported.leaves(MyList([1, 2, 3, 4]), namespace='') == [MyList([1, 2, 3, 4])]
    check_roundtrip(MyList([1, 2, 3, 4]))
    assert reexported.functools.reduce(operator.add, MyList([1, 2, 3, 4])) == 10

    registrations = reexported.register_node.get()
    global_registrations = rustree.register_pytree_node.get()
    registry = registrations[MyDataClass]
    assert registrations == rustree.register_pytree_node.get(namespace=namespace)
    assert global_registrations == reexported.register_node.get(namespace='')
    assert len(registrations) == len(global_registrations) + 2
    assert registry == rustree.register_pytree_node.get(MyDataClass, namespace=namespace)
    assert rustree.register_pytree_node.get(MyDataClass) is None
    assert registry == reexported.register_node.get(MyDataClass)
    assert reexported.register_node.get(MyDataClass, namespace='') is None


def test_pytree_reexport_without_module():
    assert importlib.import_module('test_shortcut') is sys.modules[__name__]

    assert f'{__name__}.pytree' not in sys.modules
    assert f'{__name__}.pytree.dataclasses' not in sys.modules
    assert f'{__name__}.pytree.functools' not in sys.modules

    _ = None
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree as _
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree.dataclasses as _
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import test_shortcut.pytree.functools as _
    del _

    pytree1 = rustree.pytree.reexport(namespace='pytree1')
    check_reexported_module(
        reexported=pytree1,
        module=f'{__name__}.pytree',
        namespace='pytree1',
    )

    with pytest.raises(ValueError, match=re.escape(f"module '{__name__}.pytree' already exists")):
        rustree.pytree.reexport(namespace='pytree2')


def test_pytree_reexport_with_module():
    assert 'some_package' not in sys.modules
    assert 'some_package.pytree_mod' not in sys.modules
    assert 'some_package.pytree_mod.dataclasses' not in sys.modules
    assert 'some_package.pytree_mod.functools' not in sys.modules

    sys.modules['some_package'] = types.ModuleType('some_package')

    _ = None
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod as _
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod.dataclasses as _
    with pytest.raises(ModuleNotFoundError, match=r'No module named'):
        import some_package.pytree_mod.functools as _
    del _

    pytree3 = rustree.pytree.reexport(namespace='pytree3', module='some_package.pytree_mod')
    check_reexported_module(
        reexported=pytree3,
        module='some_package.pytree_mod',
        namespace='pytree3',
    )

    with pytest.raises(
        ValueError,
        match=re.escape("module 'some_package.pytree_mod' already exists"),
    ):
        rustree.pytree.reexport(namespace='pytree4', module='some_package.pytree_mod')
