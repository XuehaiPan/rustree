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

use crate::pytypes::get_rust_module;
use crate::pytypes::{get_defaultdict, get_deque, get_ordereddict};
use crate::pytypes::{is_namedtuple_class, is_structseq_class};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::*;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::sync::{Arc, OnceLock, RwLock};

#[pyclass(eq, eq_int, module = "rustree", rename_all = "UPPERCASE")]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum PyTreeKind {
    Custom = 0,
    Leaf,
    None,
    Tuple,
    List,
    Dict,
    NamedTuple,
    OrderedDict,
    DefaultDict,
    Deque,
    StructSequence,
}

impl PyTreeKind {
    pub(crate) const NUM_KINDS: i32 = 11;

    pub(crate) fn name(self) -> &'static str {
        match self {
            PyTreeKind::Custom => "CUSTOM",
            PyTreeKind::Leaf => "LEAF",
            PyTreeKind::None => "NONE",
            PyTreeKind::Tuple => "TUPLE",
            PyTreeKind::List => "LIST",
            PyTreeKind::Dict => "DICT",
            PyTreeKind::NamedTuple => "NAMEDTUPLE",
            PyTreeKind::OrderedDict => "ORDEREDDICT",
            PyTreeKind::DefaultDict => "DEFAULTDICT",
            PyTreeKind::Deque => "DEQUE",
            PyTreeKind::StructSequence => "STRUCTSEQUENCE",
        }
    }
}

pub fn add_pytree_kind_enum(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let members = PyDict::new(py);
    for kind in [
        PyTreeKind::Custom,
        PyTreeKind::Leaf,
        PyTreeKind::None,
        PyTreeKind::Tuple,
        PyTreeKind::List,
        PyTreeKind::Dict,
        PyTreeKind::NamedTuple,
        PyTreeKind::OrderedDict,
        PyTreeKind::DefaultDict,
        PyTreeKind::Deque,
        PyTreeKind::StructSequence,
    ] {
        members.set_item(kind.name(), kind as i32)?;
    }
    let int_enum = py.import("enum")?.getattr("IntEnum")?;
    let kind_type = int_enum.call1(("PyTreeKind", members))?;
    kind_type.setattr("NUM_KINDS", PyTreeKind::NUM_KINDS)?;
    m.add("PyTreeKind", kind_type)?;
    Ok(())
}

pub fn pytree_kind_object(py: Python<'_>, kind: PyTreeKind) -> PyResult<Py<PyAny>> {
    Ok(get_rust_module(py, None)
        .bind(py)
        .getattr("PyTreeKind")?
        .call1((kind as i32,))?
        .unbind())
}

#[repr(transparent)]
struct IdHashedPy<T>(Py<T>);

impl<T> From<IdHashedPy<T>> for Py<T> {
    fn from(id_hashed_py: IdHashedPy<T>) -> Self {
        id_hashed_py.0
    }
}

impl<T> From<Py<T>> for IdHashedPy<T> {
    fn from(py: Py<T>) -> Self {
        IdHashedPy(py)
    }
}

impl<T> std::cmp::PartialEq for IdHashedPy<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}
impl<T> std::cmp::Eq for IdHashedPy<T> {}

impl<T> std::hash::Hash for IdHashedPy<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

static REGISTRY_NONE_IS_NODE: PyOnceLock<RwLock<PyTreeTypeRegistry>> = PyOnceLock::new();
static REGISTRY_NONE_IS_LEAF: PyOnceLock<RwLock<PyTreeTypeRegistry>> = PyOnceLock::new();
static DICT_INSERTION_ORDERED_NAMESPACES: OnceLock<RwLock<HashSet<String>>> = OnceLock::new();

pub(crate) struct PyTreeTypeRegistration {
    pub(crate) kind: PyTreeKind,
    pub(crate) r#type: Py<PyType>,
    pub(crate) flatten_func: Option<Py<PyAny>>,
    pub(crate) unflatten_func: Option<Py<PyAny>>,
    pub(crate) path_entry_type: Option<Py<PyType>>,
}

pub struct PyTreeTypeRegistry {
    registrations: HashMap<IdHashedPy<PyType>, Arc<PyTreeTypeRegistration>>,
    named_registrations: HashMap<(String, IdHashedPy<PyType>), Arc<PyTreeTypeRegistration>>,
    builtin_types: HashSet<IdHashedPy<PyType>>,
}

impl PyTreeTypeRegistry {
    fn new(py: Python<'_>, none_is_leaf: bool) -> RwLock<Self> {
        let init_fn = |none_is_leaf: bool| {
            move || {
                let mut singleton = PyTreeTypeRegistry {
                    registrations: HashMap::new(),
                    named_registrations: HashMap::new(),
                    builtin_types: HashSet::new(),
                };

                let mut register = |node_type: Py<PyType>, kind: PyTreeKind| {
                    singleton
                        .registrations
                        .entry(node_type.clone_ref(py).into())
                        .or_insert(Arc::new(PyTreeTypeRegistration {
                            kind,
                            r#type: node_type,
                            flatten_func: None,
                            unflatten_func: None,
                            path_entry_type: None,
                        }));
                };

                if !none_is_leaf {
                    register(py.get_type::<PyNone>().unbind(), PyTreeKind::None);
                }
                register(py.get_type::<PyTuple>().unbind(), PyTreeKind::Tuple);
                register(py.get_type::<PyList>().unbind(), PyTreeKind::List);
                register(py.get_type::<PyDict>().unbind(), PyTreeKind::Dict);
                register(get_ordereddict(py), PyTreeKind::OrderedDict);
                register(get_defaultdict(py), PyTreeKind::DefaultDict);
                register(get_deque(py), PyTreeKind::Deque);

                for hashed_type in singleton.registrations.keys() {
                    singleton
                        .builtin_types
                        .insert(hashed_type.0.clone_ref(py).into());
                }
                singleton
                    .builtin_types
                    .insert(py.get_type::<PyNone>().unbind().into());

                RwLock::new(singleton)
            }
        };

        match none_is_leaf {
            false => init_fn(false)(),
            true => init_fn(true)(),
        }
    }

    #[inline]
    fn get_singleton(py: Python<'_>, none_is_leaf: bool) -> &'static RwLock<Self> {
        match none_is_leaf {
            false => REGISTRY_NONE_IS_NODE.get_or_init(py, || Self::new(py, false)),
            true => REGISTRY_NONE_IS_LEAF.get_or_init(py, || Self::new(py, true)),
        }
    }

    #[inline]
    fn lookup_impl(
        &self,
        obj: &Bound<'_, PyAny>,
        namespace: &str,
    ) -> (PyTreeKind, Option<Arc<PyTreeTypeRegistration>>) {
        let cls = &obj.get_type();
        if !namespace.is_empty() {
            if let Some(registration) = self
                .named_registrations
                .get(&(String::from(namespace), cls.clone().unbind().into()))
            {
                return (registration.as_ref().kind, Some(registration.clone()));
            }
        }
        if let Some(registration) = self.registrations.get(&cls.clone().unbind().into()) {
            (registration.as_ref().kind, Some(registration.clone()))
        } else if is_structseq_class(cls).unwrap() {
            (PyTreeKind::StructSequence, None)
        } else if is_namedtuple_class(cls).unwrap() {
            (PyTreeKind::NamedTuple, None)
        } else {
            (PyTreeKind::Leaf, None)
        }
    }

    #[inline]
    pub fn lookup(
        obj: &Bound<'_, PyAny>,
        none_is_leaf: Option<bool>,
        namespace: Option<&str>,
    ) -> (PyTreeKind, Option<Arc<PyTreeTypeRegistration>>) {
        PyTreeTypeRegistry::get_singleton(obj.py(), none_is_leaf.unwrap_or(false))
            .read()
            .unwrap()
            .lookup_impl(obj, namespace.unwrap_or(""))
    }

    #[inline]
    pub fn lookup_type(
        cls: &Bound<'_, PyType>,
        none_is_leaf: Option<bool>,
        namespace: Option<&str>,
    ) -> Option<Arc<PyTreeTypeRegistration>> {
        let namespace = namespace.unwrap_or("");
        let key = cls.clone().unbind();
        let registry = PyTreeTypeRegistry::get_singleton(cls.py(), none_is_leaf.unwrap_or(false))
            .read()
            .unwrap();
        if !namespace.is_empty() {
            if let Some(registration) = registry
                .named_registrations
                .get(&(String::from(namespace), cls.clone().unbind().into()))
            {
                return Some(registration.clone());
            }
        }
        registry.registrations.get(&key.into()).cloned()
    }

    fn register_impl<'py>(
        &mut self,
        cls: &Bound<'py, PyType>,
        flatten_func: &Bound<'py, PyAny>,
        unflatten_func: &Bound<'py, PyAny>,
        path_entry_type: &Bound<'py, PyType>,
        namespace: &str,
    ) -> PyResult<()> {
        let py = cls.py();
        let key = IdHashedPy(cls.clone().unbind());
        if self.builtin_types.contains(&key) {
            return Err(PyValueError::new_err(std::format!(
                "PyTree type {} is a built-in type and cannot be re-registered.",
                cls.repr()?.to_cow().unwrap().as_ref()
            )));
        }
        if namespace.is_empty() {
            match self.registrations.entry(key) {
                HashMapEntry::Occupied(_) => {
                    return Err(PyValueError::new_err(std::format!(
                        "PyTree type {} is already registered in the global namespace.",
                        cls.repr()?.to_cow().unwrap().as_ref()
                    )));
                }
                HashMapEntry::Vacant(entry) => {
                    entry.insert(Arc::new(PyTreeTypeRegistration {
                        kind: PyTreeKind::Custom,
                        r#type: cls.clone().unbind(),
                        flatten_func: Some(flatten_func.clone().unbind()),
                        unflatten_func: Some(unflatten_func.clone().unbind()),
                        path_entry_type: Some(path_entry_type.clone().unbind()),
                    }));
                }
            };
            if is_structseq_class(cls)? {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    &CString::new(std::format!(
                        "PyTree type {} is a class of `PyStructSequence`, \
                        which is already registered in the global namespace. \
                        Override it with custom flatten/unflatten functions.",
                        cls.repr()?.to_cow().unwrap().as_ref()
                    ))?,
                    2,
                )?;
            } else if is_namedtuple_class(cls)? {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    &CString::new(std::format!(
                        "PyTree type {} is a subclass of `collections.namedtuple`, \
                        which is already registered in the global namespace. \
                        Override it with custom flatten/unflatten functions.",
                        cls.repr()?.to_cow().unwrap().as_ref()
                    ))?,
                    2,
                )?;
            }
        } else {
            let named_key = (String::from(namespace), key);
            match self.named_registrations.entry(named_key) {
                HashMapEntry::Occupied(_) => {
                    return Err(PyValueError::new_err(std::format!(
                        "PyTree type {} is already registered in namespace {}.",
                        cls.repr()?.to_cow().unwrap().as_ref(),
                        PyString::new(py, namespace)
                            .repr()?
                            .to_cow()
                            .unwrap()
                            .as_ref()
                    )));
                }
                HashMapEntry::Vacant(entry) => {
                    entry.insert(Arc::new(PyTreeTypeRegistration {
                        kind: PyTreeKind::Custom,
                        r#type: cls.clone().unbind(),
                        flatten_func: Some(flatten_func.clone().unbind()),
                        unflatten_func: Some(unflatten_func.clone().unbind()),
                        path_entry_type: Some(path_entry_type.clone().unbind()),
                    }));
                }
            };
            if is_structseq_class(cls)? {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    &CString::new(std::format!(
                        "PyTree type {} is a class of `PyStructSequence`, \
                        which is already registered in the global namespace. \
                        Override it with custom flatten/unflatten functions in namespace {}.",
                        cls.repr()?.to_cow().unwrap().as_ref(),
                        PyString::new(py, namespace)
                            .repr()?
                            .to_cow()
                            .unwrap()
                            .as_ref()
                    ))?,
                    2,
                )?;
            } else if is_namedtuple_class(cls)? {
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    &CString::new(std::format!(
                        "PyTree type {} is a subclass of `collections.namedtuple`, \
                        which is already registered in the global namespace. \
                        Override it with custom flatten/unflatten functions in namespace {}.",
                        cls.repr()?.to_cow().unwrap().as_ref(),
                        PyString::new(py, namespace)
                            .repr()?
                            .to_cow()
                            .unwrap()
                            .as_ref()
                    ))?,
                    2,
                )?;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn register<'py>(
        cls: &Bound<'py, PyType>,
        flatten_func: &Bound<'py, PyAny>,
        unflatten_func: &Bound<'py, PyAny>,
        path_entry_type: &Bound<'py, PyType>,
        namespace: Option<&str>,
    ) -> PyResult<()> {
        if !flatten_func.is_callable() {
            return Err(PyTypeError::new_err("'flatten_func' must be callable"));
        }
        if !unflatten_func.is_callable() {
            return Err(PyTypeError::new_err("'unflatten_func' must be callable"));
        }

        let namespace = namespace.unwrap_or("");
        PyTreeTypeRegistry::get_singleton(cls.py(), false)
            .write()
            .unwrap()
            .register_impl(
                cls,
                flatten_func,
                unflatten_func,
                path_entry_type,
                namespace,
            )?;
        PyTreeTypeRegistry::get_singleton(cls.py(), true)
            .write()
            .unwrap()
            .register_impl(
                cls,
                flatten_func,
                unflatten_func,
                path_entry_type,
                namespace,
            )?;
        Ok(())
    }

    fn unregister_impl(&mut self, cls: &Bound<'_, PyType>, namespace: &str) -> PyResult<()> {
        let py = cls.py();
        let key = IdHashedPy(cls.clone().unbind());
        if self.builtin_types.contains(&key) {
            return Err(PyValueError::new_err(std::format!(
                "PyTree type {} is a built-in type and cannot be unregistered.",
                cls.repr()?.to_cow().unwrap().as_ref()
            )));
        }
        if namespace.is_empty() {
            let registration = self.registrations.remove(&key);
            if registration.is_none() {
                let mut message = String::new();
                message.push_str("PyTree type ");
                message.push_str(cls.repr()?.to_cow().unwrap().as_ref());
                if is_structseq_class(cls)? {
                    message.push_str(
                        " is a class of `PyStructSequence`, \
                        which is not explicitly registered in the global namespace.",
                    );
                } else if is_namedtuple_class(cls)? {
                    message.push_str(
                        " is a subclass of `collections.namedtuple`, \
                        which is not explicitly registered in the global namespace.",
                    );
                } else {
                    message.push_str(" is not registered in the global namespace.");
                }
                return Err(PyValueError::new_err(message));
            }
        } else {
            let named_key = (String::from(namespace), key);
            let registration = self.named_registrations.remove(&named_key);
            if registration.is_none() {
                let mut message = String::new();
                message.push_str("PyTree type ");
                message.push_str(cls.repr()?.to_cow().unwrap().as_ref());
                if is_structseq_class(cls)? {
                    message.push_str(
                        " is a class of `PyStructSequence`, \
                        which is not explicitly registered in namespace ",
                    );
                } else if is_namedtuple_class(cls)? {
                    message.push_str(
                        " is a subclass of `collections.namedtuple`, \
                        which is not explicitly registered in namespace ",
                    );
                } else {
                    message.push_str(" is not registered in namespace ");
                }
                message.push_str(
                    PyString::new(py, namespace)
                        .repr()?
                        .to_cow()
                        .unwrap()
                        .as_ref(),
                );
                message.push('.');
                return Err(PyValueError::new_err(message));
            }
        }
        Ok(())
    }

    #[inline]
    pub fn unregister(cls: &Bound<'_, PyType>, namespace: Option<&str>) -> PyResult<()> {
        let namespace = namespace.unwrap_or("");
        PyTreeTypeRegistry::get_singleton(cls.py(), false)
            .write()
            .unwrap()
            .unregister_impl(cls, namespace)?;
        PyTreeTypeRegistry::get_singleton(cls.py(), true)
            .write()
            .unwrap()
            .unregister_impl(cls, namespace)?;
        Ok(())
    }

    #[inline]
    pub fn is_dict_insertion_ordered(
        namespace: Option<&str>,
        inherit_global_namespace: Option<bool>,
    ) -> bool {
        let namespace = namespace.unwrap_or("");
        let inherit_global_namespace = inherit_global_namespace.unwrap_or(true);
        let dict_insertion_ordered_namespaces = DICT_INSERTION_ORDERED_NAMESPACES
            .get_or_init(|| RwLock::new(HashSet::new()))
            .read()
            .unwrap();

        dict_insertion_ordered_namespaces.contains(namespace)
            || (inherit_global_namespace && dict_insertion_ordered_namespaces.contains(""))
    }

    #[inline]
    pub fn set_dict_insertion_ordered(mode: bool, namespace: Option<&str>) {
        let namespace = namespace.unwrap_or("");
        let mut dict_insertion_ordered_namespaces = DICT_INSERTION_ORDERED_NAMESPACES
            .get_or_init(|| RwLock::new(HashSet::new()))
            .write()
            .unwrap();

        if mode {
            dict_insertion_ordered_namespaces.insert(String::from(namespace));
        } else {
            dict_insertion_ordered_namespaces.remove(namespace);
        }
    }
}

#[pyfunction]
#[pyo3(signature = (cls, /, flatten_func, unflatten_func, path_entry_type, namespace=""))]
#[inline]
pub fn register_node<'py>(
    cls: &Bound<'py, PyType>,
    flatten_func: &Bound<'py, PyAny>,
    unflatten_func: &Bound<'py, PyAny>,
    path_entry_type: &Bound<'py, PyType>,
    namespace: Option<&str>,
) -> PyResult<()> {
    PyTreeTypeRegistry::register(
        cls,
        flatten_func,
        unflatten_func,
        path_entry_type,
        namespace,
    )
}

#[pyfunction]
#[pyo3(signature = (cls, /, namespace=""))]
#[inline]
pub fn unregister_node(cls: &Bound<'_, PyType>, namespace: Option<&str>) -> PyResult<()> {
    PyTreeTypeRegistry::unregister(cls, namespace)
}

#[pyfunction]
#[pyo3(signature = (namespace="", inherit_global_namespace=true))]
#[inline]
pub fn is_dict_insertion_ordered(
    namespace: Option<&str>,
    inherit_global_namespace: Option<bool>,
) -> bool {
    PyTreeTypeRegistry::is_dict_insertion_ordered(namespace, inherit_global_namespace)
}

#[pyfunction]
#[pyo3(signature = (mode, /, namespace=""))]
#[inline]
pub fn set_dict_insertion_ordered(mode: bool, namespace: Option<&str>) {
    PyTreeTypeRegistry::set_dict_insertion_ordered(mode, namespace)
}

#[pyfunction]
#[pyo3(signature = (namespace=None))]
pub fn get_registry_size(py: Python<'_>, namespace: Option<&str>) -> PyResult<usize> {
    let registry = py
        .import("rustree.registry")?
        .getattr("_NODETYPE_REGISTRY")?
        .downcast_into::<PyDict>()?;
    let namedtuple_factory = py.import("collections")?.getattr("namedtuple")?;
    let structseq = py.import("rustree.typing")?.getattr("StructSequence")?;
    let active_namespace = namespace.unwrap_or("");

    let mut total = 0usize;
    for (key, _) in registry.iter() {
        if key.eq(&namedtuple_factory)? || key.eq(&structseq)? {
            continue;
        }
        if let Ok(tuple_key) = key.downcast::<PyTuple>() {
            if tuple_key.len() == 2 {
                let key_namespace = tuple_key.get_item(0)?.extract::<String>()?;
                if namespace.is_some() && key_namespace != active_namespace {
                    continue;
                }
            }
        }
        total += 1;
    }
    Ok(total)
}
