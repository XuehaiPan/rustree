// Copyright 2024-2025 Xuehai Pan. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyTraverseError, PyVisit};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

use crate::pytypes::{get_defaultdict, get_deque, get_ordereddict, get_rust_module};
use crate::registry::{PyTreeKind, PyTreeTypeRegistration, PyTreeTypeRegistry};
use pyo3::exceptions::{
    PyIndexError, PyRecursionError, PyRuntimeError, PyTypeError, PyUserWarning, PyValueError,
};

pub const MAX_RECURSION_DEPTH: usize = 1000;

fn malformed_pickled_treespec() -> PyErr {
    PyRuntimeError::new_err("Malformed pickled PyTreeSpec.")
}

fn total_order_sort(list: &Bound<'_, PyList>) -> PyResult<()> {
    let py = list.py();
    match list.sort() {
        Ok(()) => Ok(()),
        Err(err) if err.is_instance_of::<PyTypeError>(py) => {
            let sort_key_fn = PyCFunction::new_closure(
                py,
                None,
                None,
                |args: &Bound<'_, PyTuple>,
                 _kwargs: Option<&Bound<'_, PyDict>>|
                 -> PyResult<(String, Py<PyAny>)> {
                    let obj = args.get_item(0)?;
                    let cls = obj.get_type();
                    let module = cls.getattr("__module__")?.extract::<String>()?;
                    let qualname = cls.getattr("__qualname__")?.extract::<String>()?;
                    Ok((format!("{module}.{qualname}"), obj.unbind()))
                },
            )?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("key", sort_key_fn)?;
            match list.call_method("sort", (), Some(&kwargs)) {
                Ok(_) => Ok(()),
                Err(err) if err.is_instance_of::<PyTypeError>(py) => Ok(()),
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn is_leaf(
    obj: &Bound<PyAny>,
    leaf_predicate: Option<&Bound<PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<bool> {
    if leaf_predicate.is_some() && leaf_predicate.unwrap().call1((obj,))?.is_truthy()? {
        return Ok(true);
    }
    if let (PyTreeKind::Leaf, ..) = PyTreeTypeRegistry::lookup(obj, none_is_leaf, namespace) {
        Ok(true)
    } else {
        Ok(false)
    }
}

struct Node {
    kind: PyTreeKind,
    arity: usize,
    node_data: Option<Py<PyAny>>,
    node_entries: Option<Py<PyTuple>>,
    custom: Option<Arc<PyTreeTypeRegistration>>,
    num_leaves: usize,
    num_nodes: usize,
    original_keys: Option<Py<PyList>>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            kind: self.kind,
            arity: self.arity,
            node_data: self.node_data.as_ref().map(|obj| obj.clone_ref(py)),
            node_entries: self.node_entries.as_ref().map(|obj| obj.clone_ref(py)),
            custom: self.custom.clone(),
            num_leaves: self.num_leaves,
            num_nodes: self.num_nodes,
            original_keys: self.original_keys.as_ref().map(|obj| obj.clone_ref(py)),
        })
    }
}

impl Node {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: PyTreeKind,
        arity: usize,
        node_data: Option<Py<PyAny>>,
        node_entries: Option<Py<PyTuple>>,
        custom: Option<Arc<PyTreeTypeRegistration>>,
        num_leaves: usize,
        num_nodes: usize,
        original_keys: Option<Py<PyList>>,
    ) -> Self {
        Node {
            kind,
            arity,
            node_data,
            node_entries,
            custom,
            num_leaves,
            num_nodes,
            original_keys,
        }
    }

    fn get_type(&self, py: Python) -> Py<PyAny> {
        match self.kind {
            PyTreeKind::Custom => self
                .custom
                .as_ref()
                .unwrap()
                .as_ref()
                .r#type
                .clone_ref(py)
                .into_any(),
            PyTreeKind::Leaf => PyNone::get(py).to_owned().unbind().into_any(),
            PyTreeKind::None => py.get_type::<PyNone>().unbind().into_any(),
            PyTreeKind::Tuple => py.get_type::<PyTuple>().unbind().into_any(),
            PyTreeKind::List => py.get_type::<PyList>().unbind().into_any(),
            PyTreeKind::Dict => py.get_type::<PyDict>().unbind().into_any(),
            PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                self.node_data.as_ref().unwrap().clone_ref(py)
            }
            PyTreeKind::OrderedDict => get_ordereddict(py).into_any(),
            PyTreeKind::DefaultDict => get_defaultdict(py).into_any(),
            PyTreeKind::Deque => get_deque(py).into_any(),
        }
    }

    fn make_node(&self, py: Python, children: &Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.kind {
            PyTreeKind::Leaf => {
                panic!("make_node not implemented for leaves");
            }
            PyTreeKind::None => Ok(PyNone::get(py).to_owned().unbind().into_any()),
            PyTreeKind::Tuple | PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                let tuple = PyTuple::new(py, children)?;
                if self.kind == PyTreeKind::NamedTuple {
                    let cls = self.node_data.as_ref().unwrap();
                    return Ok(cls.call(py, tuple, None)?.into_any());
                }
                if self.kind == PyTreeKind::StructSequence {
                    let cls = self.node_data.as_ref().unwrap();
                    return Ok(cls.call1(py, (tuple,))?.into_any());
                }
                Ok(tuple.unbind().into_any())
            }
            PyTreeKind::List | PyTreeKind::Deque => {
                let list = PyList::new(py, children)?;
                if self.kind == PyTreeKind::Deque {
                    let deque_type = get_deque(py);
                    let args = (list,);
                    let kwargs = [("maxlen", self.node_data.as_ref().unwrap())].into_py_dict(py)?;
                    let deque = deque_type.call(py, args, Some(&kwargs))?;
                    return Ok(deque.into_any());
                }
                Ok(list.unbind().into_any())
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                let dict = PyDict::new(py);
                if let Some(original_keys) = &self.original_keys {
                    for key in original_keys.bind(py) {
                        dict.set_item(key, PyNone::get(py))?;
                    }
                }
                let keys = if self.kind != PyTreeKind::DefaultDict {
                    self.node_data
                        .as_ref()
                        .unwrap()
                        .bind(py)
                        .downcast::<PyList>()?
                        .clone()
                } else {
                    self.node_data
                        .as_ref()
                        .unwrap()
                        .bind(py)
                        .downcast::<PyTuple>()?
                        .get_item(1)?
                        .downcast::<PyList>()?
                        .clone()
                };
                for (key, child) in keys.iter().zip(children.iter()) {
                    dict.set_item(key, child)?;
                }
                if self.kind == PyTreeKind::OrderedDict {
                    return Ok(get_ordereddict(py).call1(py, (dict,))?.into_any());
                }
                if self.kind == PyTreeKind::DefaultDict {
                    let default_factory = self
                        .node_data
                        .as_ref()
                        .unwrap()
                        .bind(py)
                        .downcast::<PyTuple>()?
                        .get_item(0)?;
                    return Ok(get_defaultdict(py)
                        .call(py, (default_factory, dict), None)?
                        .into_any());
                }
                Ok(dict.unbind().into_any())
            }
            PyTreeKind::Custom => {
                let tuple = PyTuple::new(py, children)?;
                let unflatten_func = self
                    .custom
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .unflatten_func
                    .as_ref()
                    .unwrap()
                    .bind(py)
                    .clone();
                let out = unflatten_func.call(
                    (self.node_data.as_ref().unwrap().clone_ref(py), tuple),
                    None,
                )?;
                Ok(out.unbind().into_any())
            }
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new(PyTreeKind::Leaf, 0, None, None, None, 1, 1, None)
    }
}

#[pyclass(module = "rustree", weakref)]
pub struct PyTreeSpec {
    traversal: Vec<Node>,
    none_is_leaf: bool,
    namespace: String,
}

impl Clone for PyTreeSpec {
    fn clone(&self) -> Self {
        Self::new(
            self.traversal.clone(),
            self.none_is_leaf,
            self.namespace.clone(),
        )
    }
}

impl PyTreeSpec {
    fn repr_guard() -> &'static Mutex<HashSet<(usize, thread::ThreadId)>> {
        static REPR_GUARD: OnceLock<Mutex<HashSet<(usize, thread::ThreadId)>>> = OnceLock::new();
        REPR_GUARD.get_or_init(|| Mutex::new(HashSet::new()))
    }

    fn hash_guard() -> &'static Mutex<HashSet<(usize, thread::ThreadId)>> {
        static HASH_GUARD: OnceLock<Mutex<HashSet<(usize, thread::ThreadId)>>> = OnceLock::new();
        HASH_GUARD.get_or_init(|| Mutex::new(HashSet::new()))
    }

    fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
        Ok(obj.repr()?.to_cow()?.into_owned())
    }

    fn py_string_attr(obj: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
        obj.getattr(attr)?.extract()
    }

    fn same_custom(a: &Node, b: &Node) -> bool {
        match (&a.custom, &b.custom) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    fn same_public_namespace(&self, other: &Self) -> bool {
        self.namespace.is_empty() || other.namespace.is_empty() || self.namespace == other.namespace
    }

    fn node_data_equal(py: Python<'_>, a: &Node, b: &Node) -> PyResult<bool> {
        match (&a.node_data, &b.node_data) {
            (None, None) => Ok(true),
            (Some(_), None) | (None, Some(_)) => Ok(false),
            (Some(a), Some(b)) => a.bind(py).eq(b.bind(py)),
        }
    }

    fn node_keys<'py>(&self, py: Python<'py>, node: &'py Node) -> PyResult<Bound<'py, PyList>> {
        match node.kind {
            PyTreeKind::Dict | PyTreeKind::OrderedDict => Ok(node
                .node_data
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<PyList>()?
                .clone()),
            PyTreeKind::DefaultDict => Ok(node
                .node_data
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<PyTuple>()?
                .get_item(1)?
                .downcast::<PyList>()?
                .clone()),
            _ => unreachable!("node_keys only supports mapping nodes"),
        }
    }

    fn subtree_slice<'a>(traversal: &'a [Node], root_index: usize) -> &'a [Node] {
        let root = &traversal[root_index];
        let start = root_index + 1 - root.num_nodes;
        &traversal[start..=root_index]
    }

    fn append_subtree(
        nodes: &mut Vec<Node>,
        traversal: &[Node],
        root_index: usize,
    ) -> (usize, usize) {
        let root = &traversal[root_index];
        nodes.extend(Self::subtree_slice(traversal, root_index).iter().cloned());
        (root.num_nodes, root.num_leaves)
    }

    fn mapping_key_mismatch_error(
        py: Python<'_>,
        expected_keys: &Bound<'_, PyList>,
        other_keys: &Bound<'_, PyList>,
        other_key_to_index: &Bound<'_, PyDict>,
    ) -> PyResult<PyErr> {
        let got_keys = other_keys
            .call_method0("copy")?
            .downcast::<PyList>()?
            .clone();
        total_order_sort(&got_keys)?;

        let missing_keys = PyList::empty(py);
        for key in expected_keys.iter() {
            if other_key_to_index.get_item(&key)?.is_none() {
                missing_keys.append(key)?;
            }
        }

        let expected_key_set = PyDict::new(py);
        for key in expected_keys.iter() {
            expected_key_set.set_item(&key, py.None())?;
        }

        let extra_keys = PyList::empty(py);
        for key in got_keys.iter() {
            if expected_key_set.get_item(&key)?.is_none() {
                extra_keys.append(key)?;
            }
        }

        let mut message = format!(
            "dictionary key mismatch; expected key(s): {}, got key(s): {}",
            Self::py_repr(expected_keys.as_any())?,
            Self::py_repr(got_keys.as_any())?,
        );
        if !missing_keys.is_empty() {
            message.push_str(&format!(
                ", missing key(s): {}",
                Self::py_repr(missing_keys.as_any())?,
            ));
        }
        if !extra_keys.is_empty() {
            message.push_str(&format!(
                ", extra key(s): {}",
                Self::py_repr(extra_keys.as_any())?,
            ));
        }
        message.push('.');
        Ok(PyValueError::new_err(message))
    }

    fn broadcast_to_common_suffix_impl(
        &self,
        py: Python<'_>,
        self_root_index: usize,
        other: &Self,
        other_root_index: usize,
        nodes: &mut Vec<Node>,
    ) -> PyResult<(usize, usize)> {
        let root = &self.traversal[self_root_index];
        let other_root = &other.traversal[other_root_index];

        if root.kind == PyTreeKind::Leaf {
            return Ok(Self::append_subtree(
                nodes,
                &other.traversal,
                other_root_index,
            ));
        }
        if other_root.kind == PyTreeKind::Leaf {
            return Ok(Self::append_subtree(
                nodes,
                &self.traversal,
                self_root_index,
            ));
        }
        if root.kind == PyTreeKind::None {
            if other_root.kind != PyTreeKind::None {
                return Err(PyValueError::new_err(format!(
                    "PyTreeSpecs have incompatible node types; expected type: {}, got: {}.",
                    Self::node_kind_to_string(py, root)?,
                    Self::node_kind_to_string(py, other_root)?,
                )));
            }
            nodes.push(root.clone());
            return Ok((root.num_nodes, root.num_leaves));
        }

        let mut node = root.clone();
        node.num_leaves = 0;
        node.num_nodes = 1;

        match root.kind {
            PyTreeKind::Tuple | PyTreeKind::List | PyTreeKind::Deque => {
                if root.kind != other_root.kind {
                    return Err(PyValueError::new_err(format!(
                        "PyTreeSpecs have incompatible node types; expected type: {}, got: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }
                if root.arity != other_root.arity {
                    return Err(PyValueError::new_err(format!(
                        "{} arity mismatch; expected: {}, got: {}.",
                        Self::node_kind_to_string(py, root)?,
                        root.arity,
                        other_root.arity,
                    )));
                }
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                if !matches!(
                    other_root.kind,
                    PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict
                ) {
                    return Err(PyValueError::new_err(format!(
                        "PyTreeSpecs have incompatible node types; expected type: {}, got: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }

                let expected_keys = self.node_keys(py, root)?;
                let other_keys = other.node_keys(py, other_root)?;
                let other_key_to_index = PyDict::new(py);
                for (index, key) in other_keys.iter().enumerate() {
                    other_key_to_index.set_item(key, index)?;
                }
                let keys_match = expected_keys.len() == other_keys.len()
                    && expected_keys.iter().all(|key| {
                        other_key_to_index
                            .get_item(&key)
                            .is_ok_and(|item| item.is_some())
                    });
                if !keys_match {
                    return Err(Self::mapping_key_mismatch_error(
                        py,
                        &expected_keys,
                        &other_keys,
                        &other_key_to_index,
                    )?);
                }

                let self_children = self.child_root_indices(self_root_index);
                let other_children = other.child_root_indices(other_root_index);
                for (position, key) in expected_keys.iter().enumerate() {
                    let other_position = other_key_to_index
                        .get_item(&key)?
                        .unwrap()
                        .extract::<usize>()?;
                    let (child_num_nodes, child_num_leaves) = self
                        .broadcast_to_common_suffix_impl(
                            py,
                            self_children[position],
                            other,
                            other_children[other_position],
                            nodes,
                        )?;
                    node.num_nodes += child_num_nodes;
                    node.num_leaves += child_num_leaves;
                }
                let num_nodes = node.num_nodes;
                let num_leaves = node.num_leaves;
                nodes.push(node);
                return Ok((num_nodes, num_leaves));
            }
            PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                if root.kind != other_root.kind {
                    return Err(PyValueError::new_err(format!(
                        "PyTreeSpecs have incompatible node types; expected type: {}, got: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }
                if root.arity != other_root.arity {
                    let node_name = if root.kind == PyTreeKind::NamedTuple {
                        "namedtuple"
                    } else {
                        "PyStructSequence"
                    };
                    return Err(PyValueError::new_err(format!(
                        "{node_name} arity mismatch; expected: {}, got: {}.",
                        root.arity, other_root.arity,
                    )));
                }
                if !Self::node_data_equal(py, root, other_root)? {
                    let node_name = if root.kind == PyTreeKind::NamedTuple {
                        "namedtuple"
                    } else {
                        "PyStructSequence"
                    };
                    return Err(PyValueError::new_err(format!(
                        "{node_name} type mismatch; expected type: {}, got type: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }
            }
            PyTreeKind::Custom => {
                if other_root.kind != PyTreeKind::Custom {
                    return Err(PyValueError::new_err(format!(
                        "PyTreeSpecs have incompatible node types; expected type: {}, got: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }

                let root_custom = root.custom.as_ref().unwrap();
                let other_custom = other_root.custom.as_ref().unwrap();
                if !root_custom.r#type.bind(py).is(other_custom.r#type.bind(py)) {
                    return Err(PyValueError::new_err(format!(
                        "Custom node type mismatch; expected type: {}, got type: {}.",
                        Self::node_kind_to_string(py, root)?,
                        Self::node_kind_to_string(py, other_root)?,
                    )));
                }
                if root.arity != other_root.arity {
                    return Err(PyValueError::new_err(format!(
                        "Custom type arity mismatch; expected: {}, got: {}.",
                        root.arity, other_root.arity,
                    )));
                }
                if !Self::node_data_equal(py, root, other_root)? {
                    return Err(PyValueError::new_err(format!(
                        "Mismatch custom node data; expected: {}, got: {}.",
                        Self::py_repr(root.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(other_root.node_data.as_ref().unwrap().bind(py).as_any())?,
                    )));
                }
            }
            PyTreeKind::Leaf | PyTreeKind::None => unreachable!("handled above"),
        }

        for (self_child, other_child) in self
            .child_root_indices(self_root_index)
            .into_iter()
            .zip(other.child_root_indices(other_root_index))
        {
            let (child_num_nodes, child_num_leaves) =
                self.broadcast_to_common_suffix_impl(py, self_child, other, other_child, nodes)?;
            node.num_nodes += child_num_nodes;
            node.num_leaves += child_num_leaves;
        }
        let num_nodes = node.num_nodes;
        let num_leaves = node.num_leaves;
        nodes.push(node);
        Ok((num_nodes, num_leaves))
    }

    fn broadcast_to_common_suffix_impl_root(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        if self.none_is_leaf != other.none_is_leaf {
            return Err(PyValueError::new_err(
                "PyTreeSpecs must have the same none_is_leaf value.",
            ));
        }
        if !self.namespace.is_empty()
            && !other.namespace.is_empty()
            && self.namespace != other.namespace
        {
            return Err(PyValueError::new_err(format!(
                "PyTreeSpecs must have the same namespace, got {} vs. {}.",
                Self::py_repr(PyString::new(py, &self.namespace).as_any())?,
                Self::py_repr(PyString::new(py, &other.namespace).as_any())?,
            )));
        }

        let mut traversal = Vec::with_capacity(self.traversal.len().max(other.traversal.len()));
        let (num_nodes, num_leaves) = self.broadcast_to_common_suffix_impl(
            py,
            self.root_index(),
            other,
            other.root_index(),
            &mut traversal,
        )?;
        debug_assert_eq!(num_nodes, traversal.len());
        debug_assert_eq!(
            num_leaves,
            traversal.last().map_or(0, |node| node.num_leaves)
        );

        let namespace = if other.namespace.is_empty() {
            self.namespace.clone()
        } else {
            other.namespace.clone()
        };
        Ok(Self::new(traversal, self.none_is_leaf, namespace))
    }

    fn new(traversal: Vec<Node>, none_is_leaf: bool, namespace: String) -> Self {
        PyTreeSpec {
            traversal,
            none_is_leaf,
            namespace,
        }
    }

    fn kind_from_int(value: isize) -> PyResult<PyTreeKind> {
        match value {
            0 => Ok(PyTreeKind::Custom),
            1 => Ok(PyTreeKind::Leaf),
            2 => Ok(PyTreeKind::None),
            3 => Ok(PyTreeKind::Tuple),
            4 => Ok(PyTreeKind::List),
            5 => Ok(PyTreeKind::Dict),
            6 => Ok(PyTreeKind::NamedTuple),
            7 => Ok(PyTreeKind::OrderedDict),
            8 => Ok(PyTreeKind::DefaultDict),
            9 => Ok(PyTreeKind::Deque),
            10 => Ok(PyTreeKind::StructSequence),
            _ => Err(malformed_pickled_treespec()),
        }
    }

    fn unknown_custom_type_error(py: Python<'_>, cls: &Bound<'_, PyAny>, namespace: &str) -> PyErr {
        let mut message = format!(
            "Unknown custom type in pickled PyTreeSpec: {}",
            Self::py_repr(cls).unwrap_or_else(|_| String::from("<unknown>")),
        );
        if namespace.is_empty() {
            message.push_str(" in the global namespace.");
        } else {
            message.push_str(" in namespace ");
            message.push_str(
                &Self::py_repr(PyString::new(py, namespace).as_any())
                    .unwrap_or_else(|_| format!("'{namespace}'")),
            );
            message.push('.');
        }
        PyRuntimeError::new_err(message)
    }

    fn to_pickleable(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut node_states = Vec::with_capacity(self.traversal.len());
        for node in &self.traversal {
            let node_data = node
                .node_data
                .as_ref()
                .map(|obj| obj.clone_ref(py))
                .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any());
            let node_entries = node
                .node_entries
                .as_ref()
                .map(|obj| obj.clone_ref(py).into_any())
                .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any());
            let custom_type = node
                .custom
                .as_ref()
                .map(|registration| registration.r#type.clone_ref(py).into_any())
                .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any());
            let original_keys = match node.kind {
                PyTreeKind::Dict | PyTreeKind::DefaultDict => node
                    .original_keys
                    .as_ref()
                    .map(|keys| keys.clone_ref(py).into_any())
                    .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any()),
                _ => PyNone::get(py).to_owned().unbind().into_any(),
            };
            node_states.push(PyTuple::new(
                py,
                [
                    (node.kind as isize).into_pyobject(py)?.into_any().unbind(),
                    node.arity.into_pyobject(py)?.into_any().unbind(),
                    node_data,
                    node_entries,
                    custom_type,
                    node.num_leaves.into_pyobject(py)?.into_any().unbind(),
                    node.num_nodes.into_pyobject(py)?.into_any().unbind(),
                    original_keys,
                ],
            )?);
        }
        let state_items = vec![
            PyTuple::new(py, node_states)?.unbind().into_any(),
            PyBool::new(py, self.none_is_leaf)
                .to_owned()
                .unbind()
                .into_any(),
            self.namespace
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        ];
        Ok(PyTuple::new(py, state_items)?.unbind().into_any())
    }

    fn from_pickleable(py: Python<'_>, pickleable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let state = pickleable.downcast::<PyTuple>()?;
        if state.len() != 3 {
            return Err(malformed_pickled_treespec());
        }

        let none_is_leaf = state.get_item(1)?.extract::<bool>()?;
        let namespace = state.get_item(2)?.extract::<String>()?;
        let node_states_obj = state.get_item(0)?;
        let node_states = node_states_obj.downcast::<PyTuple>()?;
        let mut traversal = Vec::with_capacity(node_states.len());

        for item in node_states.iter() {
            let t = item.downcast::<PyTuple>()?;
            let kind = Self::kind_from_int(t.get_item(0)?.extract::<isize>()?)?;
            let mut node = Node::new(kind, 0, None, None, None, 0, 0, None);
            let tuple_len = t.len();
            if tuple_len != 7 && tuple_len != 8 {
                return Err(malformed_pickled_treespec());
            }
            if tuple_len == 8 {
                let original_keys = t.get_item(7)?;
                if original_keys.is_none() {
                    if matches!(kind, PyTreeKind::Dict | PyTreeKind::DefaultDict) {
                        return Err(malformed_pickled_treespec());
                    }
                } else if matches!(kind, PyTreeKind::Dict | PyTreeKind::DefaultDict) {
                    node.original_keys = Some(original_keys.downcast::<PyList>()?.clone().unbind());
                } else {
                    return Err(malformed_pickled_treespec());
                }
            }

            node.arity = t.get_item(1)?.extract::<usize>()?;
            match kind {
                PyTreeKind::Leaf | PyTreeKind::None | PyTreeKind::Tuple | PyTreeKind::List => {
                    if !t.get_item(2)?.is_none() {
                        return Err(malformed_pickled_treespec());
                    }
                }
                PyTreeKind::Dict | PyTreeKind::OrderedDict => {
                    node.node_data = Some(
                        t.get_item(2)?
                            .downcast::<PyList>()?
                            .clone()
                            .unbind()
                            .into_any(),
                    );
                }
                PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                    node.node_data = Some(
                        t.get_item(2)?
                            .downcast::<PyType>()?
                            .clone()
                            .unbind()
                            .into_any(),
                    );
                }
                PyTreeKind::DefaultDict | PyTreeKind::Deque | PyTreeKind::Custom => {
                    node.node_data = Some(t.get_item(2)?.unbind());
                }
            }

            if kind == PyTreeKind::Custom {
                let node_entries = t.get_item(3)?;
                if !node_entries.is_none() {
                    node.node_entries = Some(node_entries.downcast::<PyTuple>()?.clone().unbind());
                }
                let custom_type = t.get_item(4)?;
                if !custom_type.is_none() {
                    let cls = custom_type.downcast::<PyType>()?;
                    node.custom =
                        PyTreeTypeRegistry::lookup_type(cls, Some(none_is_leaf), Some(&namespace));
                    if node.custom.is_none() {
                        return Err(Self::unknown_custom_type_error(
                            py,
                            cls.as_any(),
                            &namespace,
                        ));
                    }
                }
                if node.custom.is_none() {
                    return Err(Self::unknown_custom_type_error(
                        py,
                        custom_type.as_any(),
                        &namespace,
                    ));
                }
            } else if !t.get_item(3)?.is_none() || !t.get_item(4)?.is_none() {
                return Err(malformed_pickled_treespec());
            }

            node.num_leaves = t.get_item(5)?.extract::<usize>()?;
            node.num_nodes = t.get_item(6)?.extract::<usize>()?;
            traversal.push(node);
        }

        traversal.shrink_to_fit();
        Ok(Self::new(traversal, none_is_leaf, namespace))
    }

    fn node_kind_to_string(py: Python<'_>, node: &Node) -> PyResult<String> {
        Ok(match node.kind {
            PyTreeKind::Leaf => String::from("leaf type"),
            PyTreeKind::None => String::from("NoneType"),
            PyTreeKind::Tuple => String::from("tuple"),
            PyTreeKind::List => String::from("list"),
            PyTreeKind::Dict => String::from("dict"),
            PyTreeKind::OrderedDict => String::from("OrderedDict"),
            PyTreeKind::DefaultDict => String::from("defaultdict"),
            PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?
            }
            PyTreeKind::Deque => String::from("deque"),
            PyTreeKind::Custom => {
                Self::py_repr(node.custom.as_ref().unwrap().r#type.bind(py).as_any())?
            }
        })
    }

    fn warn_make_from_collection_leaf(py: Python<'_>) -> PyResult<()> {
        let warnings = py.import("warnings")?;
        let kwargs = [("stacklevel", 2)].into_py_dict(py)?;
        warnings.getattr("warn")?.call(
            (
                "PyTreeSpec::MakeFromCollection() is called on a leaf.",
                py.get_type::<PyUserWarning>(),
            ),
            Some(&kwargs),
        )?;
        Ok(())
    }

    fn make_leaf_impl(none_is_leaf: bool) -> Self {
        Self::new(
            vec![Node::new(PyTreeKind::Leaf, 0, None, None, None, 1, 1, None)],
            none_is_leaf,
            String::new(),
        )
    }

    fn make_none_impl(none_is_leaf: bool) -> Self {
        if none_is_leaf {
            return Self::make_leaf_impl(true);
        }
        Self::new(
            vec![Node::new(PyTreeKind::None, 0, None, None, None, 0, 1, None)],
            false,
            String::new(),
        )
    }

    fn verify_constructor_children(
        py: Python<'_>,
        handle: &Bound<'_, PyAny>,
        node: &Node,
        children: &[Py<PyAny>],
        none_is_leaf: bool,
        registry_namespace: &mut String,
    ) -> PyResult<Vec<Self>> {
        let mut treespecs = Vec::with_capacity(children.len());
        for child in children {
            let child_ref = child
                .bind(py)
                .extract::<PyRef<'_, PyTreeSpec>>()
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "Expected a(n) {} of PyTreeSpec(s), got {}.",
                        Self::node_kind_to_string(py, node)
                            .unwrap_or_else(|_| String::from("object")),
                        Self::py_repr(handle).unwrap_or_else(|_| String::from("<unknown>")),
                    ))
                })?;
            treespecs.push(child_ref.clone());
        }

        let mut common_registry_namespace = String::new();
        for treespec in &treespecs {
            if treespec.none_is_leaf != none_is_leaf {
                return Err(PyValueError::new_err(if none_is_leaf {
                    "Expected treespec(s) with `none_is_leaf=True`."
                } else {
                    "Expected treespec(s) with `none_is_leaf=False`."
                }));
            }
            if !treespec.namespace.is_empty() {
                if common_registry_namespace.is_empty() {
                    common_registry_namespace = treespec.namespace.clone();
                } else if common_registry_namespace != treespec.namespace {
                    return Err(PyValueError::new_err(format!(
                        "Expected treespecs with the same namespace, got {} vs. {}.",
                        Self::py_repr(PyString::new(py, &common_registry_namespace).as_any())?,
                        Self::py_repr(PyString::new(py, &treespec.namespace).as_any())?,
                    )));
                }
            }
        }

        if !common_registry_namespace.is_empty() {
            if registry_namespace.is_empty() {
                *registry_namespace = common_registry_namespace;
            } else if *registry_namespace != common_registry_namespace {
                return Err(PyValueError::new_err(format!(
                    "Expected treespec(s) with namespace {}, got {}.",
                    Self::py_repr(PyString::new(py, registry_namespace).as_any())?,
                    Self::py_repr(PyString::new(py, &common_registry_namespace).as_any())?,
                )));
            }
        } else if node.kind != PyTreeKind::Custom {
            registry_namespace.clear();
        }

        Ok(treespecs)
    }

    fn make_from_collection_impl(
        py: Python<'_>,
        handle: &Bound<'_, PyAny>,
        none_is_leaf: bool,
        namespace: &str,
    ) -> PyResult<Self> {
        let mut registry_namespace = String::from(namespace);
        let mut children = Vec::with_capacity(4);
        let mut node = Node::default();
        let (kind, registration) =
            PyTreeTypeRegistry::lookup(handle, Some(none_is_leaf), Some(namespace));
        node.kind = kind;

        match node.kind {
            PyTreeKind::Leaf => {
                Self::warn_make_from_collection_leaf(py)?;
                return Ok(Self::make_leaf_impl(none_is_leaf));
            }
            PyTreeKind::None => {
                return Ok(Self::make_none_impl(none_is_leaf));
            }
            PyTreeKind::Tuple => {
                let tuple = handle.downcast::<PyTuple>()?;
                node.arity = tuple.len();
                for child in tuple {
                    children.push(child.unbind());
                }
            }
            PyTreeKind::List => {
                let list = handle.downcast::<PyList>()?;
                node.arity = list.len();
                for child in list {
                    children.push(child.unbind());
                }
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                let dict = handle.downcast::<PyDict>()?;
                node.arity = dict.len();
                let keys = dict.keys();
                if node.kind != PyTreeKind::OrderedDict {
                    node.original_keys = Some(
                        keys.call_method0("copy")?
                            .downcast::<PyList>()?
                            .clone()
                            .unbind(),
                    );
                    if !PyTreeTypeRegistry::is_dict_insertion_ordered(
                        Some(registry_namespace.as_str()),
                        Some(true),
                    ) {
                        total_order_sort(&keys)?;
                    }
                }
                for key in keys.iter() {
                    children.push(dict.get_item(&key)?.unwrap().unbind());
                }
                if node.kind == PyTreeKind::DefaultDict {
                    let default_factory = handle.getattr("default_factory")?;
                    node.node_data = Some(
                        PyTuple::new(py, [default_factory.unbind(), keys.unbind().into_any()])?
                            .unbind()
                            .into_any(),
                    );
                } else {
                    node.node_data = Some(keys.unbind().into_any());
                }
            }
            PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                let tuple = handle.downcast::<PyTuple>()?;
                node.arity = tuple.len();
                node.node_data = Some(handle.get_type().unbind().into_any());
                for child in tuple {
                    children.push(child.unbind());
                }
            }
            PyTreeKind::Deque => {
                let list =
                    unsafe { handle.clone().downcast_into_unchecked::<PySequence>() }.to_list()?;
                node.arity = list.len();
                node.node_data = Some(handle.getattr("maxlen")?.unbind());
                for child in list {
                    children.push(child.unbind());
                }
            }
            PyTreeKind::Custom => {
                let registration = registration.unwrap();
                node.custom = Some(registration.clone());
                let flatten_func = registration.flatten_func.as_ref().unwrap().bind(py);
                let out_any = flatten_func.call1((handle,))?;
                let out = out_any.downcast::<PyTuple>()?;
                let num_out = out.len();
                if num_out != 2 && num_out != 3 {
                    return Err(PyRuntimeError::new_err(format!(
                        "PyTree custom flatten function for type {} should return a 2- or 3-tuple, got {}.",
                        Self::py_repr(registration.r#type.bind(py).as_any())?,
                        num_out,
                    )));
                }
                node.node_data = Some(out.get_item(1)?.unbind());
                for child in out.get_item(0)?.try_iter()? {
                    node.arity += 1;
                    children.push(child?.unbind());
                }
                if num_out == 3 {
                    let node_entries = out.get_item(2)?;
                    if !node_entries.is_none() {
                        let node_entries = PyTuple::new(
                            py,
                            node_entries.try_iter()?.collect::<PyResult<Vec<_>>>()?,
                        )?;
                        if node_entries.len() != node.arity {
                            return Err(PyRuntimeError::new_err(format!(
                                "PyTree custom flatten function for type {} returned inconsistent number of children ({}) and number of entries ({}).",
                                Self::py_repr(registration.r#type.bind(py).as_any())?,
                                node.arity,
                                node_entries.len(),
                            )));
                        }
                        node.node_entries = Some(node_entries.unbind());
                    }
                }
            }
        }

        let treespecs = Self::verify_constructor_children(
            py,
            handle,
            &node,
            &children,
            none_is_leaf,
            &mut registry_namespace,
        )?;
        let mut traversal = Vec::with_capacity(
            treespecs
                .iter()
                .map(|treespec| treespec.traversal.len())
                .sum::<usize>()
                + 1,
        );
        let mut num_leaves = if node.kind == PyTreeKind::Leaf { 1 } else { 0 };
        for treespec in treespecs {
            num_leaves += treespec.num_leaves()?;
            traversal.extend(treespec.traversal);
        }
        node.num_leaves = num_leaves;
        node.num_nodes = traversal.len() + 1;
        traversal.push(node);
        Ok(Self::new(traversal, none_is_leaf, registry_namespace))
    }

    fn flatten_into_impl(
        obj: &Bound<PyAny>,
        traversal: &mut Vec<Node>,
        leaves: &mut Vec<Py<PyAny>>,
        depth: usize,
        leaf_predicate: Option<&Bound<PyAny>>,
        none_is_leaf: bool,
        namespace: &str,
    ) -> PyResult<bool> {
        if depth > MAX_RECURSION_DEPTH {
            return Err(PyRecursionError::new_err(
                "Maximum recursion depth exceeded during flattening the tree.",
            ));
        }
        let mut found_custom = false;
        let start_num_nodes = traversal.len();
        let start_num_leaves = leaves.len();

        let mut node = Node::default();

        if leaf_predicate.is_some() && leaf_predicate.unwrap().call1((obj,))?.is_truthy()? {
            leaves.push(obj.clone().unbind());
        } else {
            let (kind, registration) =
                PyTreeTypeRegistry::lookup(obj, Some(none_is_leaf), Some(namespace));
            node.kind = kind;

            let mut recurse = |child| {
                Self::flatten_into_impl(
                    &child,
                    traversal,
                    leaves,
                    depth + 1,
                    leaf_predicate,
                    none_is_leaf,
                    namespace,
                )
            };

            match node.kind {
                PyTreeKind::Leaf => {
                    leaves.push(obj.clone().unbind());
                }
                PyTreeKind::None => {
                    if none_is_leaf {
                        unreachable!("None should be a leaf");
                    }
                }
                PyTreeKind::Tuple => {
                    let obj = obj.downcast::<PyTuple>()?;
                    node.arity = obj.len();
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::List => {
                    let obj = obj.downcast::<PyList>()?;
                    node.arity = obj.len();
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                    let obj = obj.downcast::<PyDict>()?;
                    node.arity = obj.len();
                    let keys = obj.keys();
                    let preserve_insertion_order = node.kind == PyTreeKind::OrderedDict
                        || (matches!(node.kind, PyTreeKind::Dict | PyTreeKind::DefaultDict)
                            && PyTreeTypeRegistry::is_dict_insertion_ordered(
                                Some(namespace),
                                Some(true),
                            ));
                    node.original_keys = Some(
                        keys.call_method0("copy")?
                            .downcast::<PyList>()?
                            .as_unbound()
                            .clone_ref(obj.py()),
                    );
                    if !preserve_insertion_order {
                        total_order_sort(&keys)?;
                    }
                    for key in &keys {
                        let child = obj.get_item(key)?.unwrap();
                        found_custom |= recurse(child.clone())?;
                    }
                    if node.kind == PyTreeKind::DefaultDict {
                        let default_factory = obj.getattr("default_factory")?;
                        node.node_data = Some(
                            PyTuple::new(obj.py(), &[default_factory, keys.into_any()])?
                                .unbind()
                                .into_any(),
                        );
                    } else {
                        node.node_data = Some(keys.unbind().into_any());
                    }
                }
                PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                    let obj = obj.downcast::<PyTuple>()?;
                    node.arity = obj.len();
                    node.node_data = Some(obj.get_type().unbind().into_any());
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::Deque => {
                    let list =
                        unsafe { obj.clone().downcast_into_unchecked::<PySequence>() }.to_list()?;
                    node.arity = list.len();
                    for child in list {
                        found_custom |= recurse(child)?;
                    }
                    node.node_data = Some(obj.getattr("maxlen")?.unbind());
                }
                PyTreeKind::Custom => {
                    found_custom = true;
                    let registration = registration.unwrap();
                    node.custom = Some(registration.clone());
                    let flatten_func = registration
                        .flatten_func
                        .as_ref()
                        .unwrap()
                        .bind(obj.py())
                        .clone();
                    let out_any = flatten_func.call1((obj,))?;
                    let out = out_any.downcast::<PyTuple>()?;
                    let num_out = out.len();
                    if num_out != 2 && num_out != 3 {
                        return Err(PyRuntimeError::new_err(format!(
                            "PyTree custom flatten function for type {} should return a 2- or 3-tuple, got {}.",
                            Self::py_repr(registration.r#type.bind(obj.py()).as_any())?,
                            num_out,
                        )));
                    }
                    node.node_data = Some(out.get_item(1)?.unbind());
                    let children = out.get_item(0)?;
                    for child in children.try_iter()? {
                        found_custom |= recurse(child?)?;
                        node.arity += 1;
                    }
                    if num_out == 3 {
                        let node_entries = out.get_item(2)?;
                        if !node_entries.is_none() {
                            let node_entries = PyTuple::new(
                                obj.py(),
                                node_entries.try_iter()?.collect::<PyResult<Vec<_>>>()?,
                            )?;
                            if node_entries.len() != node.arity {
                                return Err(PyRuntimeError::new_err(format!(
                                    "PyTree custom flatten function for type {} returned inconsistent number of children ({}) and number of entries ({}).",
                                    Self::py_repr(registration.r#type.bind(obj.py()).as_any())?,
                                    node.arity,
                                    node_entries.len(),
                                )));
                            }
                            node.node_entries = Some(node_entries.unbind());
                        }
                    }
                }
            }
        }

        node.num_leaves = leaves.len() - start_num_leaves;
        node.num_nodes = traversal.len() - start_num_nodes + 1;
        traversal.push(node);
        Ok(found_custom)
    }

    fn flatten(
        obj: &Bound<PyAny>,
        leaf_predicate: Option<&Bound<PyAny>>,
        none_is_leaf: bool,
        namespace: &str,
    ) -> PyResult<(Vec<Py<PyAny>>, PyTreeSpec)> {
        let mut traversal = Vec::with_capacity(4);
        let mut leaves = Vec::with_capacity(4);
        let found_custom = Self::flatten_into_impl(
            obj,
            &mut traversal,
            &mut leaves,
            0,
            leaf_predicate,
            none_is_leaf,
            namespace,
        )?;

        traversal.shrink_to_fit();
        let namespace = if found_custom
            || PyTreeTypeRegistry::is_dict_insertion_ordered(Some(namespace), Some(false))
        {
            String::from(namespace)
        } else {
            String::from("")
        };

        let treespec = PyTreeSpec::new(traversal, none_is_leaf, namespace);
        Ok((leaves, treespec))
    }

    fn root_index(&self) -> usize {
        self.traversal.len() - 1
    }

    fn child_root_indices(&self, root_index: usize) -> Vec<usize> {
        let node = &self.traversal[root_index];
        let mut child_roots = Vec::with_capacity(node.arity);
        if node.arity == 0 {
            return child_roots;
        }

        let mut cursor = root_index - 1;
        for _ in 0..node.arity {
            child_roots.push(cursor);
            cursor = cursor.saturating_sub(self.traversal[cursor].num_nodes);
        }
        child_roots.reverse();
        child_roots
    }

    fn node_entries<'py>(&self, py: Python<'py>, node: &Node) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(entries) = &node.node_entries {
            return Ok(entries.bind(py).clone());
        }

        let make_range = || -> PyResult<Bound<'py, PyTuple>> {
            Ok(PyTuple::new(py, (0..node.arity).collect::<Vec<_>>())?)
        };

        match node.kind {
            PyTreeKind::Leaf | PyTreeKind::None => Ok(PyTuple::empty(py)),
            PyTreeKind::Tuple
            | PyTreeKind::List
            | PyTreeKind::Deque
            | PyTreeKind::NamedTuple
            | PyTreeKind::StructSequence
            | PyTreeKind::Custom => make_range(),
            PyTreeKind::Dict | PyTreeKind::OrderedDict => Ok(node
                .node_data
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<PyList>()?
                .to_tuple()),
            PyTreeKind::DefaultDict => Ok(node
                .node_data
                .as_ref()
                .unwrap()
                .bind(py)
                .downcast::<PyTuple>()?
                .get_item(1)?
                .downcast::<PyList>()?
                .to_tuple()),
        }
    }

    fn node_path_entry_type<'py>(
        &self,
        py: Python<'py>,
        node: &Node,
    ) -> PyResult<Bound<'py, PyAny>> {
        let accessors = py.import("rustree.accessors")?;
        match node.kind {
            PyTreeKind::Tuple | PyTreeKind::List | PyTreeKind::Deque => {
                accessors.getattr("SequenceEntry")
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                accessors.getattr("MappingEntry")
            }
            PyTreeKind::NamedTuple => accessors.getattr("NamedTupleEntry"),
            PyTreeKind::StructSequence => accessors.getattr("StructSequenceEntry"),
            PyTreeKind::Custom => Ok(node
                .custom
                .as_ref()
                .and_then(|registration| registration.path_entry_type.as_ref())
                .map(|tp| tp.bind(py).clone().into_any())
                .unwrap_or_else(|| accessors.getattr("AutoEntry").unwrap())),
            PyTreeKind::Leaf | PyTreeKind::None => accessors.getattr("PyTreeEntry"),
        }
    }

    fn subtree(&self, root_index: usize) -> Self {
        let node = &self.traversal[root_index];
        let start = root_index + 1 - node.num_nodes;
        Self::new(
            self.traversal[start..=root_index].to_vec(),
            self.none_is_leaf,
            self.namespace.clone(),
        )
    }

    fn one_level_from_node(&self, node: &Node) -> Self {
        if node.kind == PyTreeKind::Leaf {
            return Self::new(
                vec![Node::default()],
                self.none_is_leaf,
                self.namespace.clone(),
            );
        }

        let mut traversal = std::iter::repeat_with(Node::default)
            .take(node.arity)
            .collect::<Vec<_>>();
        let mut root = node.clone();
        root.num_leaves = node.arity;
        root.num_nodes = node.arity + 1;
        traversal.push(root);
        Self::new(traversal, self.none_is_leaf, self.namespace.clone())
    }

    fn collect_paths(
        &self,
        py: Python<'_>,
        root_index: usize,
        prefix: &mut Vec<Py<PyAny>>,
        out: &mut Vec<Py<PyTuple>>,
    ) -> PyResult<()> {
        let node = &self.traversal[root_index];
        if node.kind == PyTreeKind::Leaf {
            out.push(PyTuple::new(py, prefix.iter().map(|obj| obj.bind(py)))?.unbind());
            return Ok(());
        }
        if node.arity == 0 {
            return Ok(());
        }

        let entries = self.node_entries(py, node)?;
        for (entry, child_root) in entries.iter().zip(self.child_root_indices(root_index)) {
            prefix.push(entry.unbind());
            self.collect_paths(py, child_root, prefix, out)?;
            prefix.pop();
        }
        Ok(())
    }

    fn collect_accessors(
        &self,
        py: Python<'_>,
        root_index: usize,
        prefix: &mut Vec<Py<PyAny>>,
        out: &mut Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let node = &self.traversal[root_index];
        if node.kind == PyTreeKind::Leaf {
            let accessor = py
                .import("rustree.accessors")?
                .getattr("PyTreeAccessor")?
                .call1((PyTuple::new(py, prefix.iter().map(|obj| obj.bind(py)))?,))?;
            out.push(accessor.unbind());
            return Ok(());
        }
        if node.arity == 0 {
            return Ok(());
        }

        let entry_type = self.node_path_entry_type(py, node)?;
        let node_type = node.get_type(py);
        let node_kind = crate::registry::pytree_kind_object(py, node.kind)?;
        let entries = self.node_entries(py, node)?;
        for (entry, child_root) in entries.iter().zip(self.child_root_indices(root_index)) {
            let path_entry = entry_type.call1((entry, node_type.bind(py), node_kind.bind(py)))?;
            prefix.push(path_entry.unbind());
            self.collect_accessors(py, child_root, prefix, out)?;
            prefix.pop();
        }
        Ok(())
    }

    fn to_string_impl(&self, py: Python<'_>) -> PyResult<String> {
        let mut agenda = Vec::with_capacity(self.traversal.len().max(1));
        for node in &self.traversal {
            let child_start = agenda.len() - node.arity;
            let children_slice = &agenda[child_start..];
            let children = children_slice.join(", ");

            let representation = match node.kind {
                PyTreeKind::Leaf => String::from("*"),
                PyTreeKind::None => String::from("None"),
                PyTreeKind::Tuple => {
                    let trailing_comma = if node.arity == 1 { "," } else { "" };
                    format!("({children}{trailing_comma})")
                }
                PyTreeKind::List => format!("[{children}]"),
                PyTreeKind::Dict | PyTreeKind::OrderedDict => {
                    let keys = node
                        .node_data
                        .as_ref()
                        .unwrap()
                        .bind(py)
                        .downcast::<PyList>()?;
                    let mut items = Vec::with_capacity(node.arity);
                    for (key, child) in keys.iter().zip(children_slice.iter()) {
                        items.push(format!("{}: {}", Self::py_repr(&key)?, child));
                    }
                    if node.kind == PyTreeKind::OrderedDict {
                        if items.is_empty() {
                            String::from("OrderedDict()")
                        } else {
                            format!("OrderedDict({{{}}})", items.join(", "))
                        }
                    } else {
                        format!("{{{}}}", items.join(", "))
                    }
                }
                PyTreeKind::NamedTuple => {
                    let ty = node.node_data.as_ref().unwrap().bind(py);
                    let fields_obj = ty.getattr("_fields")?;
                    let fields = fields_obj.downcast::<PyTuple>()?;
                    let name = Self::py_string_attr(ty.as_any(), "__name__")?;
                    let mut items = Vec::with_capacity(node.arity);
                    for (field, child) in fields.iter().zip(children_slice.iter()) {
                        items.push(format!("{}={}", field.extract::<String>()?, child));
                    }
                    format!("{name}({})", items.join(", "))
                }
                PyTreeKind::DefaultDict => {
                    let metadata = node
                        .node_data
                        .as_ref()
                        .unwrap()
                        .bind(py)
                        .downcast::<PyTuple>()?;
                    let default_factory = metadata.get_item(0)?;
                    let keys_obj = metadata.get_item(1)?;
                    let keys = keys_obj.downcast::<PyList>()?;
                    let mut items = Vec::with_capacity(node.arity);
                    for (key, child) in keys.iter().zip(children_slice.iter()) {
                        items.push(format!("{}: {}", Self::py_repr(&key)?, child));
                    }
                    format!(
                        "defaultdict({}, {{{}}})",
                        Self::py_repr(&default_factory)?,
                        items.join(", ")
                    )
                }
                PyTreeKind::Deque => {
                    let maxlen = node.node_data.as_ref().unwrap().bind(py);
                    if maxlen.is_none() {
                        format!("deque([{children}])")
                    } else {
                        format!(
                            "deque([{children}], maxlen={})",
                            Self::py_repr(maxlen.as_any())?
                        )
                    }
                }
                PyTreeKind::StructSequence => {
                    let ty = node.node_data.as_ref().unwrap().bind(py);
                    let fields = crate::pytypes::structseq_fields(ty.as_any())?;
                    let mut prefix = String::new();
                    let module_name = ty.getattr("__module__")?;
                    if !module_name.is_none() {
                        let module_name = module_name.extract::<String>()?;
                        if !module_name.is_empty()
                            && module_name != "__main__"
                            && module_name != "builtins"
                            && module_name != "__builtins__"
                        {
                            prefix.push_str(&module_name);
                            prefix.push('.');
                        }
                    }
                    prefix.push_str(&Self::py_string_attr(ty.as_any(), "__qualname__")?);

                    let mut items = Vec::with_capacity(node.arity);
                    for (field, child) in fields.iter().zip(children_slice.iter()) {
                        items.push(format!("{}={}", field.extract::<String>()?, child));
                    }
                    format!("{prefix}({})", items.join(", "))
                }
                PyTreeKind::Custom => {
                    let custom = node.custom.as_ref().unwrap();
                    let name = Self::py_string_attr(custom.r#type.bind(py).as_any(), "__name__")?;
                    let metadata = node
                        .node_data
                        .as_ref()
                        .map(|data| Self::py_repr(data.bind(py).as_any()))
                        .transpose()?
                        .unwrap_or_default();
                    format!("CustomTreeNode({name}[{metadata}], [{children}])")
                }
            };

            agenda.truncate(child_start);
            agenda.push(representation);
        }

        let mut out = format!("PyTreeSpec({})", agenda.last().cloned().unwrap_or_default());
        if self.none_is_leaf {
            out.pop();
            out.push_str(", NoneIsLeaf)");
        }
        if !self.namespace.is_empty() {
            out.pop();
            out.push_str(", namespace=");
            out.push_str(&Self::py_repr(PyString::new(py, &self.namespace).as_any())?);
            out.push(')');
        }
        Ok(out)
    }

    fn to_string(&self, py: Python<'_>) -> PyResult<String> {
        let ident = (self as *const Self as usize, thread::current().id());
        {
            let guard = Self::repr_guard().lock().unwrap();
            if guard.contains(&ident) {
                return Ok(String::from("..."));
            }
        }

        struct ReprReset {
            ident: (usize, thread::ThreadId),
        }
        impl Drop for ReprReset {
            fn drop(&mut self) {
                PyTreeSpec::repr_guard().lock().unwrap().remove(&self.ident);
            }
        }

        Self::repr_guard().lock().unwrap().insert(ident.clone());
        let _reset = ReprReset { ident };
        self.to_string_impl(py)
    }

    fn equal_to(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        if self.traversal.len() != other.traversal.len()
            || self.none_is_leaf != other.none_is_leaf
            || !self.same_public_namespace(other)
            || self.num_nodes()? != other.num_nodes()?
            || self.num_leaves()? != other.num_leaves()?
        {
            return Ok(false);
        }

        for (a, b) in self.traversal.iter().zip(other.traversal.iter()) {
            if a.kind != b.kind
                || a.arity != b.arity
                || a.node_data.is_some() != b.node_data.is_some()
                || !Self::same_custom(a, b)
                || !Self::node_data_equal(py, a, b)?
                || a.num_leaves != b.num_leaves
                || a.num_nodes != b.num_nodes
            {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn prefix_match(
        &self,
        py: Python<'_>,
        self_index: usize,
        other: &Self,
        other_index: usize,
    ) -> PyResult<(bool, bool)> {
        let a = &self.traversal[self_index];
        let b = &other.traversal[other_index];

        if a.kind == PyTreeKind::Leaf {
            return Ok((true, b.kind == PyTreeKind::Leaf));
        }

        if a.arity != b.arity
            || a.node_data.is_some() != b.node_data.is_some()
            || !Self::same_custom(a, b)
            || a.num_nodes > b.num_nodes
        {
            return Ok((false, false));
        }

        match a.kind {
            PyTreeKind::None | PyTreeKind::Tuple | PyTreeKind::List | PyTreeKind::Deque => {
                if a.kind != b.kind {
                    return Ok((false, false));
                }
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                if !matches!(
                    b.kind,
                    PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict
                ) {
                    return Ok((false, false));
                }
            }
            PyTreeKind::NamedTuple | PyTreeKind::StructSequence | PyTreeKind::Custom => {
                if a.kind != b.kind || !Self::node_data_equal(py, a, b)? {
                    return Ok((false, false));
                }
            }
            PyTreeKind::Leaf => unreachable!("leaf handled above"),
        }

        let mut all_leaves_match = true;
        match a.kind {
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                let expected_keys = self.node_keys(py, a)?;
                let other_keys = other.node_keys(py, b)?;
                if expected_keys.len() != other_keys.len() {
                    return Ok((false, false));
                }

                let key_to_index = PyDict::new(py);
                for (index, key) in other_keys.iter().enumerate() {
                    key_to_index.set_item(key, index)?;
                }

                let self_children = self.child_root_indices(self_index);
                let other_children = other.child_root_indices(other_index);
                for (position, key) in expected_keys.iter().enumerate() {
                    let Some(index) = key_to_index.get_item(&key)? else {
                        return Ok((false, false));
                    };
                    let other_child = other_children[index.extract::<usize>()?];
                    let (matched, leaves_match) =
                        self.prefix_match(py, self_children[position], other, other_child)?;
                    if !matched {
                        return Ok((false, false));
                    }
                    all_leaves_match &= leaves_match;
                }
            }
            PyTreeKind::Leaf => unreachable!("leaf handled above"),
            _ => {
                for (child_a, child_b) in self
                    .child_root_indices(self_index)
                    .into_iter()
                    .zip(other.child_root_indices(other_index))
                {
                    let (matched, leaves_match) = self.prefix_match(py, child_a, other, child_b)?;
                    if !matched {
                        return Ok((false, false));
                    }
                    all_leaves_match &= leaves_match;
                }
            }
        }

        Ok((true, all_leaves_match))
    }

    fn is_prefix_of(&self, py: Python<'_>, other: &Self, strict: bool) -> PyResult<bool> {
        if self.none_is_leaf != other.none_is_leaf
            || !self.same_public_namespace(other)
            || self.traversal.len() > other.traversal.len()
        {
            return Ok(false);
        }

        let (matched, all_leaves_match) =
            self.prefix_match(py, self.root_index(), other, other.root_index())?;
        Ok(matched && (!strict || !all_leaves_match))
    }

    fn hash_value(&self, py: Python<'_>) -> PyResult<isize> {
        let ident = (self as *const Self as usize, thread::current().id());
        {
            let guard = Self::hash_guard().lock().unwrap();
            if guard.contains(&ident) {
                return Ok(0);
            }
        }

        struct HashReset {
            ident: (usize, thread::ThreadId),
        }
        impl Drop for HashReset {
            fn drop(&mut self) {
                PyTreeSpec::hash_guard().lock().unwrap().remove(&self.ident);
            }
        }

        Self::hash_guard().lock().unwrap().insert(ident.clone());
        let _reset = HashReset { ident };
        PyString::new(py, &self.to_string(py)?).hash()
    }

    fn transform_impl(
        &self,
        py: Python<'_>,
        f_node: Option<&Bound<'_, PyAny>>,
        f_leaf: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if f_node.is_none() && f_leaf.is_none() {
            return Ok(self.clone());
        }

        let mut traversal = Vec::with_capacity(self.traversal.len().max(1));
        let mut common_namespace = self.namespace.clone();
        let mut num_extra_leaves = 0usize;
        let mut num_extra_nodes = 0usize;
        let mut pending_num_leaves_nodes = Vec::with_capacity(4);

        for node in &self.traversal {
            let input_spec = self.one_level_from_node(node);
            let func = if node.kind == PyTreeKind::Leaf {
                f_leaf
            } else {
                f_node
            };

            let transformed = if let Some(func) = func {
                let out = func.call1((input_spec.clone(),))?;
                let transformed_ref = out.extract::<PyRef<'_, PyTreeSpec>>().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "Expected the PyTreeSpec transform function returns a PyTreeSpec, got {} (input: {}).",
                        Self::py_repr(&out).unwrap_or_else(|_| String::from("<unknown>")),
                        input_spec.to_string(py).unwrap_or_else(|_| String::from("PyTreeSpec(?)")),
                    ))
                })?;
                transformed_ref.clone()
            } else {
                input_spec
            };

            if transformed.none_is_leaf != self.none_is_leaf {
                return Err(PyValueError::new_err(format!(
                    "Expected the PyTreeSpec transform function returns a PyTreeSpec with the same value of {} as the input, got {} (input: {}).",
                    if self.none_is_leaf {
                        "`none_is_leaf=True`"
                    } else {
                        "`none_is_leaf=False`"
                    },
                    transformed.to_string(py)?,
                    self.one_level_from_node(node).to_string(py)?,
                )));
            }
            if !transformed.namespace.is_empty() {
                if common_namespace.is_empty() {
                    common_namespace = transformed.namespace.clone();
                } else if transformed.namespace != common_namespace {
                    return Err(PyValueError::new_err(format!(
                        "Expected the PyTreeSpec transform function returns a PyTreeSpec with namespace {}, got {}.",
                        Self::py_repr(PyString::new(py, &common_namespace).as_any())?,
                        Self::py_repr(PyString::new(py, &transformed.namespace).as_any())?,
                    )));
                }
            }

            if node.kind != PyTreeKind::Leaf {
                if transformed.num_leaves()? != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "Expected the PyTreeSpec transform function returns a PyTreeSpec with the same number of arity as the input ({}), got {} (input: {}).",
                        node.arity,
                        transformed.to_string(py)?,
                        self.one_level_from_node(node).to_string(py)?,
                    )));
                }
                if transformed.num_nodes()? != node.arity + 1 {
                    return Err(PyValueError::new_err(format!(
                        "Expected the PyTreeSpec transform function returns a one-level PyTreeSpec as the input, got {} (input: {}).",
                        transformed.to_string(py)?,
                        self.one_level_from_node(node).to_string(py)?,
                    )));
                }

                let mut subroot = transformed.traversal.last().unwrap().clone();
                subroot.num_leaves = 0;
                subroot.num_nodes = 1;
                for _ in 0..node.arity {
                    let (num_leaves, num_nodes) = pending_num_leaves_nodes.pop().unwrap();
                    subroot.num_leaves += num_leaves;
                    subroot.num_nodes += num_nodes;
                }
                pending_num_leaves_nodes.push((subroot.num_leaves, subroot.num_nodes));
                traversal.push(subroot);
            } else {
                traversal.extend(transformed.traversal.clone());
                let num_leaves = transformed.num_leaves()?;
                let num_nodes = transformed.num_nodes()?;
                num_extra_leaves += num_leaves.saturating_sub(1);
                num_extra_nodes += num_nodes.saturating_sub(1);
                pending_num_leaves_nodes.push((num_leaves, num_nodes));
            }
        }

        let root = traversal.last().unwrap();
        debug_assert_eq!(pending_num_leaves_nodes.len(), 1);
        debug_assert_eq!(root.num_leaves, self.num_leaves()? + num_extra_leaves);
        debug_assert_eq!(root.num_nodes, self.num_nodes()? + num_extra_nodes);
        Ok(Self::new(traversal, self.none_is_leaf, common_namespace))
    }

    fn compose_impl(&self, py: Python<'_>, inner: &Self) -> PyResult<Self> {
        if self.none_is_leaf != inner.none_is_leaf {
            return Err(PyValueError::new_err(
                "PyTreeSpecs must have the same none_is_leaf value.",
            ));
        }
        if !self.namespace.is_empty()
            && !inner.namespace.is_empty()
            && self.namespace != inner.namespace
        {
            return Err(PyValueError::new_err(format!(
                "PyTreeSpecs must have the same namespace, got {} vs. {}.",
                Self::py_repr(PyString::new(py, &self.namespace).as_any())?,
                Self::py_repr(PyString::new(py, &inner.namespace).as_any())?,
            )));
        }

        let mut traversal = Vec::with_capacity(
            (self.num_nodes()? - self.num_leaves()? + (self.num_leaves()? * inner.num_nodes()?))
                .max(1),
        );
        for node in &self.traversal {
            if node.kind == PyTreeKind::Leaf {
                traversal.extend(inner.traversal.clone());
            } else {
                let mut new_node = node.clone();
                new_node.num_leaves = node.num_leaves * inner.num_leaves()?;
                new_node.num_nodes =
                    (node.num_nodes - node.num_leaves) + (node.num_leaves * inner.num_nodes()?);
                traversal.push(new_node);
            }
        }
        Ok(Self::new(
            traversal,
            self.none_is_leaf,
            if inner.namespace.is_empty() {
                self.namespace.clone()
            } else {
                inner.namespace.clone()
            },
        ))
    }

    fn walk_impl(
        &self,
        py: Python<'_>,
        leaves: &Bound<'_, PyAny>,
        f_node: Option<&Bound<'_, PyAny>>,
        f_leaf: Option<&Bound<'_, PyAny>>,
        pass_raw_node: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut agenda = Vec::with_capacity(4);
        let mut it = leaves.try_iter()?;

        for node in &self.traversal {
            match node.kind {
                PyTreeKind::Leaf => {
                    let leaf = match it.next() {
                        Some(Ok(leaf)) => leaf,
                        Some(Err(err)) => return Err(err),
                        None => {
                            return Err(PyValueError::new_err("Too few leaves for PyTreeSpec."));
                        }
                    };
                    let out = if let Some(f_leaf) = f_leaf {
                        f_leaf.call1((leaf,))?.unbind()
                    } else {
                        leaf.unbind()
                    };
                    agenda.push(out);
                }
                _ => {
                    let size = agenda.len();
                    debug_assert!(size >= node.arity);
                    if pass_raw_node && f_node.is_some() {
                        let children = agenda.split_off(size - node.arity);
                        let children =
                            PyTuple::new(py, children.iter().map(|child| child.bind(py)))?;
                        let node_data = node
                            .node_data
                            .as_ref()
                            .map(|obj| obj.clone_ref(py))
                            .unwrap_or_else(|| PyNone::get(py).to_owned().unbind().into_any());
                        let out = f_node
                            .unwrap()
                            .call1((node.get_type(py), node_data, children))?
                            .unbind();
                        agenda.push(out);
                    } else {
                        let children = agenda.split_off(size - node.arity);
                        let out = node.make_node(py, &children)?;
                        let out = if let Some(f_node) = f_node {
                            f_node.call1((out.bind(py),))?.unbind()
                        } else {
                            out
                        };
                        agenda.push(out);
                    }
                }
            }
        }

        match it.next() {
            Some(Ok(_)) => Err(PyValueError::new_err("Too many leaves for PyTreeSpec.")),
            Some(Err(err)) => Err(err),
            None => Ok(agenda.pop().unwrap()),
        }
    }

    fn flatten_up_to_node(
        &self,
        py: Python<'_>,
        root_index: usize,
        object: &Bound<'_, PyAny>,
        out: &mut Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let node = &self.traversal[root_index];
        match node.kind {
            PyTreeKind::Leaf => {
                out.push(object.clone().unbind());
            }
            PyTreeKind::None => {
                if !object.is_none() {
                    return Err(PyValueError::new_err(format!(
                        "Expected None, got {}.",
                        Self::py_repr(object)?
                    )));
                }
            }
            PyTreeKind::Tuple => {
                if !object.is_exact_instance_of::<PyTuple>() {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of tuple, got {}.",
                        Self::py_repr(object)?
                    )));
                }
                let tuple = object.downcast::<PyTuple>()?;
                if tuple.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "tuple arity mismatch; expected: {}, got: {}; tuple: {}.",
                        node.arity,
                        tuple.len(),
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(tuple.iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::List => {
                if !object.is_exact_instance_of::<PyList>() {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of list, got {}.",
                        Self::py_repr(object)?
                    )));
                }
                let list = object.downcast::<PyList>()?;
                if list.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "list arity mismatch; expected: {}, got: {}; list: {}.",
                        node.arity,
                        list.len(),
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(list.iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                let (object_kind, _) = PyTreeTypeRegistry::lookup(
                    object,
                    Some(self.none_is_leaf),
                    Some(self.namespace.as_str()),
                );
                if !matches!(
                    object_kind,
                    PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict
                ) {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of dict, collections.OrderedDict, or collections.defaultdict, got {}.",
                        Self::py_repr(object)?
                    )));
                }
                let dict = object.downcast::<PyDict>()?;
                let expected_keys = self.node_keys(py, node)?;
                let got_keys = dict.keys();
                if got_keys.len() != expected_keys.len()
                    || !expected_keys
                        .iter()
                        .all(|key| dict.contains(&key).unwrap_or(false))
                {
                    let mut sorted_keys = got_keys;
                    total_order_sort(&sorted_keys)?;
                    let dict_name = match object_kind {
                        PyTreeKind::OrderedDict => "OrderedDict",
                        PyTreeKind::DefaultDict => "defaultdict",
                        _ => "dict",
                    };
                    return Err(PyValueError::new_err(format!(
                        "dictionary key mismatch; expected key(s): {}, got key(s): {}; {}: {}.",
                        Self::py_repr(expected_keys.as_any())?,
                        Self::py_repr(sorted_keys.as_any())?,
                        dict_name,
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, key) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(expected_keys.iter())
                {
                    let child = dict.get_item(&key)?.unwrap();
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::NamedTuple => {
                if !crate::pytypes::is_namedtuple_instance(object)? {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of {}, got {}.",
                        Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(object)?
                    )));
                }
                let tuple = object.downcast::<PyTuple>()?;
                if tuple.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "namedtuple arity mismatch; expected: {}, got: {}; tuple: {}.",
                        node.arity,
                        tuple.len(),
                        Self::py_repr(object)?
                    )));
                }
                if !object
                    .get_type()
                    .eq(node.node_data.as_ref().unwrap().bind(py))?
                {
                    return Err(PyValueError::new_err(format!(
                        "namedtuple type mismatch; expected type: {}, got type: {}; tuple: {}.",
                        Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(object.get_type().as_any())?,
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(tuple.iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::Deque => {
                if !object.get_type().eq(get_deque(py).bind(py))? {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of collections.deque, got {}.",
                        Self::py_repr(object)?
                    )));
                }
                let list =
                    unsafe { object.clone().downcast_into_unchecked::<PySequence>() }.to_list()?;
                if list.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "deque arity mismatch; expected: {}, got: {}; deque: {}.",
                        node.arity,
                        list.len(),
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(list.iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::StructSequence => {
                if !crate::pytypes::is_structseq_instance(object)? {
                    return Err(PyValueError::new_err(format!(
                        "Expected an instance of {}, got {}.",
                        Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(object)?
                    )));
                }
                let tuple = object.downcast::<PyTuple>()?;
                if tuple.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "PyStructSequence arity mismatch; expected: {}, got: {}; tuple: {}.",
                        node.arity,
                        tuple.len(),
                        Self::py_repr(object)?
                    )));
                }
                if !object
                    .get_type()
                    .eq(node.node_data.as_ref().unwrap().bind(py))?
                {
                    return Err(PyValueError::new_err(format!(
                        "PyStructSequence type mismatch; expected type: {}, got type: {}; tuple: {}.",
                        Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(object.get_type().as_any())?,
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(tuple.iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
            PyTreeKind::Custom => {
                let (object_kind, registration) = PyTreeTypeRegistry::lookup(
                    object,
                    Some(self.none_is_leaf),
                    Some(self.namespace.as_str()),
                );
                if object_kind != PyTreeKind::Custom
                    || !Self::same_custom(
                        node,
                        &Node {
                            kind: object_kind,
                            arity: 0,
                            node_data: None,
                            node_entries: None,
                            custom: registration.clone(),
                            num_leaves: 0,
                            num_nodes: 0,
                            original_keys: None,
                        },
                    )
                {
                    return Err(PyValueError::new_err(format!(
                        "Custom node type mismatch; expected type: {}, got type: {}; value: {}.",
                        Self::py_repr(node.custom.as_ref().unwrap().r#type.bind(py).as_any())?,
                        Self::py_repr(object.get_type().as_any())?,
                        Self::py_repr(object)?
                    )));
                }

                let registration = registration.unwrap();
                let flatten_func = registration.flatten_func.as_ref().unwrap().bind(py);
                let out_any = flatten_func.call1((object,))?;
                let out_tuple = out_any.downcast::<PyTuple>()?;
                let num_out = out_tuple.len();
                if num_out != 2 && num_out != 3 {
                    return Err(PyRuntimeError::new_err(format!(
                        "PyTree custom flatten function for type {} should return a 2- or 3-tuple, got {}.",
                        Self::py_repr(registration.r#type.bind(py).as_any())?,
                        num_out,
                    )));
                }
                let node_data = out_tuple.get_item(1)?;
                if !node_data.eq(node.node_data.as_ref().unwrap().bind(py))? {
                    return Err(PyValueError::new_err(format!(
                        "Mismatch custom node data; expected: {}, got: {}; value: {}.",
                        Self::py_repr(node.node_data.as_ref().unwrap().bind(py).as_any())?,
                        Self::py_repr(&node_data)?,
                        Self::py_repr(object)?
                    )));
                }
                let children = out_tuple.get_item(0)?;
                let children = children.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                if children.len() != node.arity {
                    return Err(PyValueError::new_err(format!(
                        "Custom type arity mismatch; expected: {}, got: {}; value: {}.",
                        node.arity,
                        children.len(),
                        Self::py_repr(object)?
                    )));
                }
                for (child_root, child) in self
                    .child_root_indices(root_index)
                    .into_iter()
                    .zip(children.into_iter())
                {
                    self.flatten_up_to_node(py, child_root, &child, out)?;
                }
            }
        }
        Ok(())
    }

    fn flatten_up_to_impl(
        &self,
        py: Python<'_>,
        tree: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut out = Vec::with_capacity(self.num_leaves()?);
        self.flatten_up_to_node(py, self.root_index(), tree, &mut out)?;
        Ok(out)
    }
}

#[pyclass(module = "rustree", weakref)]
pub struct PyTreeIter {
    root: Option<Py<PyAny>>,
    agenda: Mutex<Vec<(Py<PyAny>, usize)>>,
    leaf_predicate: Option<Py<PyAny>>,
    none_is_leaf: bool,
    namespace: String,
    is_dict_insertion_ordered: bool,
}

impl PyTreeIter {
    fn next_impl(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut agenda = self.agenda.lock().unwrap();
        while let Some((object, depth)) = agenda.pop() {
            if depth > MAX_RECURSION_DEPTH {
                return Err(PyRecursionError::new_err(
                    "Maximum recursion depth exceeded during flattening the tree.",
                ));
            }

            let bound = object.bind(py);
            if let Some(leaf_predicate) = &self.leaf_predicate {
                if leaf_predicate.bind(py).call1((bound,))?.is_truthy()? {
                    return Ok(object);
                }
            }

            let (kind, registration) = PyTreeTypeRegistry::lookup(
                bound,
                Some(self.none_is_leaf),
                Some(self.namespace.as_str()),
            );
            let next_depth = depth + 1;
            match kind {
                PyTreeKind::Leaf => return Ok(object),
                PyTreeKind::None => {}
                PyTreeKind::Tuple => {
                    let tuple = bound.downcast::<PyTuple>()?;
                    for child in tuple.iter().rev() {
                        agenda.push((child.unbind(), next_depth));
                    }
                }
                PyTreeKind::List => {
                    let list = bound.downcast::<PyList>()?;
                    for child in list.iter().rev() {
                        agenda.push((child.unbind(), next_depth));
                    }
                }
                PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                    let dict = bound.downcast::<PyDict>()?;
                    let keys = dict.keys();
                    if kind != PyTreeKind::OrderedDict && !self.is_dict_insertion_ordered {
                        total_order_sort(&keys)?;
                    }
                    for key in keys.iter().rev() {
                        let child = dict.get_item(key)?.unwrap();
                        agenda.push((child.unbind(), next_depth));
                    }
                }
                PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                    let tuple = bound.downcast::<PyTuple>()?;
                    for child in tuple.iter().rev() {
                        agenda.push((child.unbind(), next_depth));
                    }
                }
                PyTreeKind::Deque => {
                    let list = unsafe { bound.clone().downcast_into_unchecked::<PySequence>() }
                        .to_list()?;
                    for child in list.iter().rev() {
                        agenda.push((child.unbind(), next_depth));
                    }
                }
                PyTreeKind::Custom => {
                    let registration = registration.unwrap();
                    let flatten_func = registration.flatten_func.as_ref().unwrap().bind(py);
                    let out_any = flatten_func.call1((bound,))?;
                    let out = out_any.downcast::<PyTuple>()?;
                    let num_out = out.len();
                    if num_out != 2 && num_out != 3 {
                        return Err(PyRuntimeError::new_err(format!(
                            "PyTree custom flatten function for type {} should return a 2- or 3-tuple, got {}.",
                            PyTreeSpec::py_repr(registration.r#type.bind(py).as_any())?,
                            num_out,
                        )));
                    }
                    let children = out.get_item(0)?.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                    if num_out == 3 {
                        let node_entries = out.get_item(2)?;
                        if !node_entries.is_none() {
                            let num_entries = node_entries.try_iter()?.count();
                            if num_entries != children.len() {
                                return Err(PyRuntimeError::new_err(format!(
                                    "PyTree custom flatten function for type {} returned inconsistent number of children ({}) and number of entries ({}).",
                                    PyTreeSpec::py_repr(registration.r#type.bind(py).as_any())?,
                                    children.len(),
                                    num_entries,
                                )));
                            }
                        }
                    }
                    for child in children.into_iter().rev() {
                        agenda.push((child.unbind(), next_depth));
                    }
                }
            }
        }

        Err(PyStopIteration::new_err(""))
    }
}

#[pyfunction]
pub fn _deserialize_treespec(
    py: Python<'_>,
    pickleable: &Bound<'_, PyAny>,
) -> PyResult<PyTreeSpec> {
    PyTreeSpec::from_pickleable(py, pickleable)
}

#[pyfunction(name = "_pytreespec_traverse")]
#[pyo3(signature = (treespec, leaves, /, f_node=None, f_leaf=None))]
pub fn pytreespec_apply(
    py: Python<'_>,
    treespec: &PyTreeSpec,
    leaves: &Bound<'_, PyAny>,
    f_node: Option<&Bound<'_, PyAny>>,
    f_leaf: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    treespec.walk_impl(py, leaves, f_node, f_leaf, false)
}

#[pyfunction]
#[pyo3(signature = (none_is_leaf=false, namespace=""))]
pub fn make_leaf(none_is_leaf: Option<bool>, namespace: Option<&str>) -> PyResult<PyTreeSpec> {
    let _ = namespace;
    Ok(PyTreeSpec::make_leaf_impl(none_is_leaf.unwrap_or(false)))
}

#[pyfunction]
#[pyo3(signature = (none_is_leaf=false, namespace=""))]
pub fn make_none(none_is_leaf: Option<bool>, namespace: Option<&str>) -> PyResult<PyTreeSpec> {
    let _ = namespace;
    Ok(PyTreeSpec::make_none_impl(none_is_leaf.unwrap_or(false)))
}

#[pyfunction]
#[pyo3(signature = (collection, /, none_is_leaf=false, namespace=""))]
pub fn make_from_collection(
    py: Python<'_>,
    collection: &Bound<'_, PyAny>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<PyTreeSpec> {
    PyTreeSpec::make_from_collection_impl(
        py,
        collection,
        none_is_leaf.unwrap_or(false),
        namespace.unwrap_or(""),
    )
}

#[pymethods]
impl PyTreeIter {
    #[new]
    #[pyo3(signature = (tree, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
    fn new(
        tree: &Bound<'_, PyAny>,
        leaf_predicate: Option<&Bound<'_, PyAny>>,
        none_is_leaf: Option<bool>,
        namespace: Option<&str>,
    ) -> Self {
        let none_is_leaf = none_is_leaf.unwrap_or(false);
        let namespace = namespace.unwrap_or("");
        Self {
            root: Some(tree.clone().unbind()),
            agenda: Mutex::new(vec![(tree.clone().unbind(), 0)]),
            leaf_predicate: leaf_predicate.map(|predicate| predicate.clone().unbind()),
            none_is_leaf,
            namespace: String::from(namespace),
            is_dict_insertion_ordered: PyTreeTypeRegistry::is_dict_insertion_ordered(
                Some(namespace),
                Some(true),
            ),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyTreeIter> {
        slf.into()
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.next_impl(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.root.as_ref())?;
        visit.call(self.leaf_predicate.as_ref())?;
        let agenda = self.agenda.lock().unwrap();
        for (object, _) in agenda.iter() {
            visit.call(Some(object))?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.root = None;
        self.leaf_predicate = None;
        self.agenda.get_mut().unwrap().clear();
    }
}

#[pymethods]
impl PyTreeSpec {
    #[getter]
    #[inline]
    fn num_leaves(&self) -> PyResult<usize> {
        Ok(self.traversal.last().unwrap().num_leaves)
    }

    #[getter]
    #[inline]
    fn num_nodes(&self) -> PyResult<usize> {
        Ok(self.traversal.len())
    }

    #[getter]
    #[inline]
    fn num_children(&self) -> PyResult<usize> {
        Ok(self.traversal.last().unwrap().arity)
    }

    #[getter]
    #[inline]
    fn none_is_leaf(&self) -> PyResult<bool> {
        Ok(self.none_is_leaf)
    }

    #[getter]
    #[inline]
    fn namespace(&self) -> PyResult<String> {
        Ok(self.namespace.clone())
    }

    #[getter]
    #[inline]
    fn r#type(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(self.traversal.last().unwrap().get_type(py))
    }

    #[getter]
    #[inline]
    fn kind(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::registry::pytree_kind_object(py, self.traversal.last().unwrap().kind)
    }

    #[inline]
    fn unflatten(&self, py: Python<'_>, leaves: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mut agenda = Vec::with_capacity(4);
        let mut num_leaves = 0;
        let mut it = leaves.try_iter()?;
        for node in self.traversal.iter() {
            if node.kind == PyTreeKind::Leaf {
                match it.next() {
                    Some(Ok(leaf)) => {
                        agenda.push(leaf.clone().unbind());
                        num_leaves += 1;
                    }
                    Some(Err(e)) => {
                        return Err(e);
                    }
                    None => {
                        return Err(PyValueError::new_err(format!(
                            "Too few leaves for PyTreeSpec; expected: {}, got: {}.",
                            Self::num_leaves(&self).unwrap(),
                            num_leaves,
                        )));
                    }
                }
            } else {
                let size = agenda.len();
                let obj = node.make_node(py, &agenda.split_off(size - node.arity))?;
                agenda.push(obj);
            }
        }
        match it.next() {
            Some(Ok(_)) => {
                return Err(PyValueError::new_err(format!(
                    "Too many leaves for PyTreeSpec; expected: {}.",
                    Self::num_leaves(&self).unwrap(),
                )));
            }
            Some(Err(e)) => {
                return Err(e);
            }
            None => {}
        }
        if agenda.len() != 1 {
            panic!("PyTreeSpec traversal did not yield a singleton.");
        }
        Ok(agenda.pop().unwrap())
    }

    #[inline]
    fn entries(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        Ok(self
            .node_entries(py, &self.traversal[self.root_index()])?
            .iter()
            .map(|entry| entry.unbind())
            .collect())
    }

    fn entry(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let entries = self.node_entries(py, &self.traversal[self.root_index()])?;
        let len = entries.len() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err(
                "PyTreeSpec::Entry() index out of range.",
            ));
        }
        Ok(entries.get_item(index as usize)?.unbind())
    }

    fn children(&self) -> Vec<PyTreeSpec> {
        self.child_root_indices(self.root_index())
            .into_iter()
            .map(|index| self.subtree(index))
            .collect()
    }

    fn child(&self, index: isize) -> PyResult<PyTreeSpec> {
        let child_roots = self.child_root_indices(self.root_index());
        let len = child_roots.len() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err(
                "PyTreeSpec::Child() index out of range.",
            ));
        }
        Ok(self.subtree(child_roots[index as usize]))
    }

    fn paths(&self, py: Python<'_>) -> PyResult<Vec<Py<PyTuple>>> {
        let mut prefix = Vec::new();
        let mut out = Vec::new();
        self.collect_paths(py, self.root_index(), &mut prefix, &mut out)?;
        Ok(out)
    }

    fn accessors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut prefix = Vec::new();
        let mut out = Vec::new();
        self.collect_accessors(py, self.root_index(), &mut prefix, &mut out)?;
        Ok(out)
    }

    fn flatten_up_to(&self, py: Python<'_>, tree: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        self.flatten_up_to_impl(py, tree)
    }

    fn broadcast_to_common_suffix(
        &self,
        py: Python<'_>,
        other: &PyTreeSpec,
    ) -> PyResult<PyTreeSpec> {
        self.broadcast_to_common_suffix_impl_root(py, other)
    }

    #[pyo3(signature = (f_node=None, f_leaf=None))]
    fn transform(
        &self,
        py: Python<'_>,
        f_node: Option<&Bound<'_, PyAny>>,
        f_leaf: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTreeSpec> {
        self.transform_impl(py, f_node, f_leaf)
    }

    fn compose(&self, py: Python<'_>, inner: &PyTreeSpec) -> PyResult<PyTreeSpec> {
        self.compose_impl(py, inner)
    }

    #[pyo3(signature = (leaves, /, f_node=None, f_leaf=None))]
    fn walk(
        &self,
        py: Python<'_>,
        leaves: &Bound<'_, PyAny>,
        f_node: Option<&Bound<'_, PyAny>>,
        f_leaf: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.walk_impl(py, leaves, f_node, f_leaf, true)
    }

    #[pyo3(signature = (*, strict=true))]
    fn is_leaf(&self, strict: bool) -> bool {
        if strict {
            self.traversal[self.root_index()].kind == PyTreeKind::Leaf
        } else {
            self.traversal.len() == 1
        }
    }

    fn is_one_level(&self) -> bool {
        let root = &self.traversal[self.root_index()];
        root.kind != PyTreeKind::Leaf
            && root.kind != PyTreeKind::None
            && self
                .child_root_indices(self.root_index())
                .into_iter()
                .all(|index| self.traversal[index].kind == PyTreeKind::Leaf)
    }

    fn one_level(&self) -> Option<PyTreeSpec> {
        let root = &self.traversal[self.root_index()];
        if root.kind == PyTreeKind::Leaf {
            return None;
        }
        Some(self.one_level_from_node(root))
    }

    fn __copy__(&self) -> PyTreeSpec {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyTreeSpec {
        self.clone()
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for node in &self.traversal {
            visit.call(node.node_data.as_ref())?;
            visit.call(node.node_entries.as_ref())?;
            visit.call(node.original_keys.as_ref())?;
            if let Some(custom) = &node.custom {
                visit.call(Some(&custom.r#type))?;
                visit.call(custom.flatten_func.as_ref())?;
                visit.call(custom.unflatten_func.as_ref())?;
                visit.call(custom.path_entry_type.as_ref())?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.traversal.clear();
        self.namespace.clear();
    }

    fn __len__(&self) -> usize {
        self.traversal[self.root_index()].num_leaves
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.to_string(py)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.to_string(py)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.hash_value(py)
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_pickleable(py)
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyAny>,))> {
        let deserialize = get_rust_module(py, None)
            .bind(py)
            .getattr("_deserialize_treespec")?
            .unbind();
        Ok((deserialize, (self.to_pickleable(py)?,)))
    }

    #[pyo3(signature = (other, /, *, strict=false))]
    fn is_prefix(&self, py: Python<'_>, other: &PyTreeSpec, strict: bool) -> PyResult<bool> {
        self.is_prefix_of(py, other, strict)
    }

    #[pyo3(signature = (other, /, *, strict=false))]
    fn is_suffix(&self, py: Python<'_>, other: &PyTreeSpec, strict: bool) -> PyResult<bool> {
        other.is_prefix_of(py, self, strict)
    }

    fn __richcmp__(&self, py: Python<'_>, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => self.equal_to(py, other),
            CompareOp::Ne => Ok(!self.equal_to(py, other)?),
            CompareOp::Lt => self.is_prefix_of(py, other, true),
            CompareOp::Le => self.is_prefix_of(py, other, false),
            CompareOp::Gt => other.is_prefix_of(py, self, true),
            CompareOp::Ge => other.is_prefix_of(py, self, false),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn flatten(
    obj: &Bound<PyAny>,
    leaf_predicate: Option<&Bound<PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<(Vec<Py<PyAny>>, PyTreeSpec)> {
    let none_is_leaf = none_is_leaf.unwrap_or(false);
    let namespace = namespace.unwrap_or("");
    PyTreeSpec::flatten(obj, leaf_predicate, none_is_leaf, namespace)
}

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn flatten_with_path(
    obj: &Bound<PyAny>,
    leaf_predicate: Option<&Bound<PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<(Vec<Py<PyTuple>>, Vec<Py<PyAny>>, PyTreeSpec)> {
    let none_is_leaf = none_is_leaf.unwrap_or(false);
    let namespace = namespace.unwrap_or("");
    let (leaves, treespec) = PyTreeSpec::flatten(obj, leaf_predicate, none_is_leaf, namespace)?;
    let paths = treespec.paths(obj.py())?;
    Ok((paths, leaves, treespec))
}
