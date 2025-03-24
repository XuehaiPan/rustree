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

use pyo3::prelude::*;
use pyo3::types::*;
use std::sync::Arc;

use crate::pytypes::{get_defaultdict, get_deque, get_ordereddict};
use crate::registry::{PyTreeKind, PyTreeTypeRegistration, PyTreeTypeRegistry};
use pyo3::exceptions::{PyRecursionError, PyValueError};

pub const MAX_RECURSION_DEPTH: usize = 1000;

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
        Self::new(PyTreeKind::Leaf, 0, None, None, None, 0, 0, None)
    }
}

#[pyclass(module = "rustree")]
pub struct PyTreeSpec {
    traversal: Vec<Node>,
    none_is_leaf: bool,
    namespace: String,
}

impl PyTreeSpec {
    fn new(traversal: Vec<Node>, none_is_leaf: bool, namespace: String) -> Self {
        PyTreeSpec {
            traversal,
            none_is_leaf,
            namespace,
        }
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
                    node.original_keys = Some(
                        keys.call_method0("copy")?
                            .downcast::<PyList>()?
                            .as_unbound()
                            .clone_ref(obj.py()),
                    );
                    if node.kind != PyTreeKind::OrderedDict {
                        keys.sort()?;
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
                    let flatten_func = registration
                        .unwrap()
                        .flatten_func
                        .as_ref()
                        .unwrap()
                        .bind(obj.py())
                        .clone();
                    let out = flatten_func.call1((obj,))?;
                    let out = unsafe { out.downcast_into_unchecked::<PySequence>() }.to_tuple()?;
                    if out.len() != 2 && out.len() != 3 {
                        return Err(PyValueError::new_err(
                            "Custom flatten function must return a tuple of length 2 or 3.",
                        ));
                    }
                    node.node_data = Some(out.get_item(1)?.unbind());
                    let children = out.get_item(0)?;
                    for child in children.try_iter()? {
                        found_custom |= recurse(child?)?;
                        node.arity += 1;
                    }
                    if out.len() == 3 {
                        let node_entries = out.get_item(2)?;
                        if !node_entries.is_none() {
                            let node_entries =
                                unsafe { node_entries.downcast_into_unchecked::<PySequence>() }
                                    .to_tuple()?;
                            if node_entries.len() != node.arity {
                                return Err(PyValueError::new_err(
                                    "Custom flatten function must return a tuple of length 3, where the third element is a tuple of the same length as the first element.",
                                ));
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
        let namespace = if found_custom {
            String::from(namespace)
        } else {
            String::from("")
        };

        let treespec = PyTreeSpec::new(traversal, none_is_leaf, namespace);
        Ok((leaves, treespec))
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
    fn kind(&self) -> PyResult<PyTreeKind> {
        Ok(self.traversal.last().unwrap().kind)
    }

    #[inline]
    fn unflatten(&self, py: Python<'_>, leaves: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let mut agenda = Vec::with_capacity(4);
        let mut num_leaves = 0;
        let mut it = leaves.try_iter()?;
        for node in self.traversal.iter() {
            if node.kind == PyTreeKind::Leaf {
                match it.next() {
                    Some(leaf) => {
                        agenda.push(leaf.unwrap().clone().unbind());
                        num_leaves += 1;
                    }
                    None => {
                        panic!("found {}", num_leaves);
                    }
                }
            } else {
                let size = agenda.len();
                let mut children = Vec::with_capacity(node.arity);
                for _ in 0..node.arity {
                    match agenda.pop() {
                        Some(child) => {
                            children.push(child);
                        }
                        None => {
                            panic!("found {}", num_leaves);
                        }
                    };
                }
                children.reverse();
                let obj = node.make_node(py, &children)?;
                agenda.truncate(size - node.arity);
                agenda.push(obj);
            }
        }
        if agenda.len() != 1 {
            panic!("found {}", num_leaves);
        }
        Ok(agenda.pop().unwrap())
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
