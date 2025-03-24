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

use crate::rustree::pytypes::{get_defaultdict, get_deque, get_ordereddict};
use crate::rustree::registry::{PyTreeKind, PyTreeTypeRegistration};

pub struct Node {
    pub kind: PyTreeKind,
    pub arity: usize,
    pub node_data: Option<Py<PyAny>>,
    pub node_entries: Option<Py<PyTuple>>,
    pub custom: Option<Arc<PyTreeTypeRegistration>>,
    pub num_leaves: usize,
    pub num_nodes: usize,
    pub original_keys: Option<Py<PyList>>,
}

impl Node {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    pub fn new(traversal: Vec<Node>, none_is_leaf: bool, namespace: String) -> Self {
        PyTreeSpec {
            traversal,
            none_is_leaf,
            namespace,
        }
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
    pub fn unflatten(&self, py: Python<'_>, leaves: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
