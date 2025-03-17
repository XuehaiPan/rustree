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
use pyo3::sync::GILOnceCell;
use pyo3::types::*;
use std::collections::{HashMap, HashSet};

#[pyclass(eq, eq_int, module = "rustree")]
#[derive(PartialEq)]
pub enum PyTreeKind {
    CUSTOM = 0,
    LEAF,
    NONE,
    TUPLE,
    LIST,
    DICT,
    NAMEDTUPLE,
    ORDEREDDICT,
    DEFAULTDICT,
    DEQUE,
    STRUCTSEQUENCE,
}

#[repr(transparent)]
struct IdHashedPy<T>(Py<T>);

impl<T> std::cmp::PartialEq for IdHashedPy<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}
impl<T> std::cmp::Eq for IdHashedPy<T> {}

impl<T> std::hash::Hash for IdHashedPy<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

static REGISTRY: GILOnceCell<PyTreeTypeRegistry> = GILOnceCell::new();
static BUILTIN_TYPES: GILOnceCell<HashSet<IdHashedPy<PyAny>>> = GILOnceCell::new();

pub struct PyTreeTypeRegistration {
    kind: PyTreeKind,
    type_: Py<PyAny>,
    flatten_func: Option<Py<PyFunction>>,
    unflatten_func: Option<Py<PyFunction>>,
    path_entry_type: Option<Py<PyAny>>,
}

pub struct PyTreeTypeRegistry {
    registrations: HashMap<IdHashedPy<PyAny>, PyTreeTypeRegistration>,
    named_registrations: HashMap<(String, IdHashedPy<PyAny>), PyTreeTypeRegistration>,
}

impl PyTreeTypeRegistry {
    fn new(py: Python<'_>) -> &'static Self {
        REGISTRY.get_or_init(py, || {
            let mut singleton = PyTreeTypeRegistry {
                registrations: HashMap::new(),
                named_registrations: HashMap::new(),
            };
            let collections = py.import("collections").unwrap();
            let ordereddict = collections.getattr("OrderedDict").unwrap();
            let defaultdict = collections.getattr("defaultdict").unwrap();
            let deque = collections.getattr("deque").unwrap();

            for (type_, kind) in [
                (py.get_type::<PyNone>().into_any(), PyTreeKind::NONE),
                (py.get_type::<PyTuple>().into_any(), PyTreeKind::TUPLE),
                (py.get_type::<PyList>().into_any(), PyTreeKind::LIST),
                (py.get_type::<PyDict>().into_any(), PyTreeKind::DICT),
                (ordereddict, PyTreeKind::ORDEREDDICT),
                (defaultdict, PyTreeKind::DEFAULTDICT),
                (deque, PyTreeKind::DEQUE),
            ] {
                let type_ = type_.unbind();
                singleton
                    .registrations
                    .entry(IdHashedPy(type_.clone_ref(py)))
                    .or_insert(PyTreeTypeRegistration {
                        kind,
                        type_,
                        flatten_func: None,
                        unflatten_func: None,
                        path_entry_type: None,
                    });
            }

            singleton
        })
    }

    pub fn get_singleton(py: Python<'_>) -> &'static Self {
        Self::new(py)
    }
}

impl Drop for PyTreeTypeRegistry {
    fn drop(&mut self) {
        Python::with_gil(|_py| {
            self.registrations.clear();
            self.named_registrations.clear();
        })
    }
}

pub fn registry_lookup<'py, 's>(
    cls: &Bound<'py, PyAny>,
    namespace: Option<&str>,
) -> Option<&'s PyTreeTypeRegistration> {
    let registry = PyTreeTypeRegistry::get_singleton(cls.py());

    let namespace = namespace.unwrap_or("");
    if !namespace.is_empty() {
        match registry
            .named_registrations
            .get(&(String::from(namespace), IdHashedPy(cls.clone().unbind())))
        {
            Some(registration) => return Some(registration),
            _ => (),
        }
    }
    registry
        .registrations
        .get(&IdHashedPy(cls.clone().unbind()))
}
