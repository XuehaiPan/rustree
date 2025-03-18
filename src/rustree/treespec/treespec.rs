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
}

#[pyclass(module = "rustree")]
pub struct PyTreeSpec {
    traversal: Vec<Node>,
    none_is_leaf: bool,
    namespace: String,
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
}
