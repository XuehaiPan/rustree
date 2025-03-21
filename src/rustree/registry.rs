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

use crate::rustree::pytypes::{is_namedtuple_class, is_structseq_class};
use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::*;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;

#[pyclass(eq, eq_int, module = "rustree", rename_all = "UPPERCASE")]
#[derive(PartialEq, Eq, Clone, Copy)]
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

static mut REGISTRY_NONE_IS_NODE: GILOnceCell<PyTreeTypeRegistry> = GILOnceCell::new();
static mut REGISTRY_NONE_IS_LEAF: GILOnceCell<PyTreeTypeRegistry> = GILOnceCell::new();
static mut DICT_INSERTION_ORDERED_NAMESPACES: OnceCell<HashSet<String>> = OnceCell::new();

pub struct PyTreeTypeRegistration {
    kind: PyTreeKind,
    node_type: Py<PyType>,
    flatten_func: Option<Py<PyAny>>,
    unflatten_func: Option<Py<PyAny>>,
    path_entry_type: Option<Py<PyType>>,
}

pub struct PyTreeTypeRegistry {
    registrations: HashMap<IdHashedPy<PyType>, PyTreeTypeRegistration>,
    named_registrations: HashMap<(String, IdHashedPy<PyType>), PyTreeTypeRegistration>,
    builtin_types: HashSet<IdHashedPy<PyType>>,
}

impl PyTreeTypeRegistry {
    fn new(py: Python<'_>, none_is_leaf: bool) -> &'static mut Self {
        let init_fn = |none_is_leaf: bool| {
            move || {
                let mut singleton = PyTreeTypeRegistry {
                    registrations: HashMap::new(),
                    named_registrations: HashMap::new(),
                    builtin_types: HashSet::new(),
                };
                let collections = py.import("collections").unwrap();
                let ordereddict = collections.getattr("OrderedDict").unwrap();
                let defaultdict = collections.getattr("defaultdict").unwrap();
                let deque = collections.getattr("deque").unwrap();
                let ordereddict = ordereddict.extract::<Bound<PyType>>().unwrap();
                let defaultdict = defaultdict.extract::<Bound<PyType>>().unwrap();
                let deque = deque.extract::<Bound<PyType>>().unwrap();

                let mut register = |node_type: Py<PyType>, kind: PyTreeKind| {
                    singleton
                        .registrations
                        .entry(node_type.clone_ref(py).into())
                        .or_insert(PyTreeTypeRegistration {
                            kind,
                            node_type: node_type.clone_ref(py),
                            flatten_func: None,
                            unflatten_func: None,
                            path_entry_type: None,
                        });
                };

                if none_is_leaf {
                    register(py.get_type::<PyNone>().unbind(), PyTreeKind::Leaf);
                }
                register(py.get_type::<PyTuple>().unbind(), PyTreeKind::Tuple);
                register(py.get_type::<PyList>().unbind(), PyTreeKind::List);
                register(py.get_type::<PyDict>().unbind(), PyTreeKind::Dict);
                register(ordereddict.unbind(), PyTreeKind::OrderedDict);
                register(defaultdict.unbind(), PyTreeKind::DefaultDict);
                register(deque.unbind(), PyTreeKind::Deque);

                for type_ in singleton.registrations.keys() {
                    singleton.builtin_types.insert(type_.0.clone_ref(py).into());
                }
                singleton
                    .builtin_types
                    .insert(py.get_type::<PyNone>().unbind().into());

                singleton
            }
        };

        #[allow(static_mut_refs)]
        match none_is_leaf {
            false => unsafe { REGISTRY_NONE_IS_NODE.get_or_init(py, init_fn(false)) },
            true => unsafe { REGISTRY_NONE_IS_LEAF.get_or_init(py, init_fn(true)) },
        };

        #[allow(static_mut_refs)]
        match none_is_leaf {
            false => unsafe { REGISTRY_NONE_IS_NODE.get_mut() }.unwrap(),
            true => unsafe { REGISTRY_NONE_IS_LEAF.get_mut() }.unwrap(),
        }
    }

    #[inline]
    fn get_singleton(py: Python<'_>, none_is_leaf: bool) -> &'static mut Self {
        Self::new(py, none_is_leaf)
    }

    #[inline]
    fn lookup_impl(
        &'static self,
        cls: &Bound<'_, PyType>,
        namespace: &str,
    ) -> Option<&'static PyTreeTypeRegistration> {
        if !namespace.is_empty() {
            if let Some(registration) = self
                .named_registrations
                .get(&(String::from(namespace), cls.clone().unbind().into()))
            {
                return Some(registration);
            }
        }
        self.registrations.get(&cls.clone().unbind().into())
    }

    #[inline]
    pub fn lookup(
        cls: &Bound<'_, PyType>,
        none_is_leaf: Option<bool>,
        namespace: Option<&str>,
    ) -> Option<&'static PyTreeTypeRegistration> {
        PyTreeTypeRegistry::get_singleton(cls.py(), none_is_leaf.unwrap_or(false))
            .lookup_impl(cls, namespace.unwrap_or(""))
    }

    fn register_impl<'py>(
        &'static mut self,
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
                    entry.insert(PyTreeTypeRegistration {
                        kind: PyTreeKind::Custom,
                        node_type: cls.clone().unbind(),
                        flatten_func: Some(flatten_func.clone().unbind()),
                        unflatten_func: Some(unflatten_func.clone().unbind()),
                        path_entry_type: Some(path_entry_type.clone().unbind()),
                    });
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
                    entry.insert(PyTreeTypeRegistration {
                        kind: PyTreeKind::Custom,
                        node_type: cls.clone().unbind(),
                        flatten_func: Some(flatten_func.clone().unbind()),
                        unflatten_func: Some(unflatten_func.clone().unbind()),
                        path_entry_type: Some(path_entry_type.clone().unbind()),
                    });
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
        PyTreeTypeRegistry::get_singleton(cls.py(), false).register_impl(
            cls,
            flatten_func,
            unflatten_func,
            path_entry_type,
            namespace,
        )?;
        PyTreeTypeRegistry::get_singleton(cls.py(), true).register_impl(
            cls,
            flatten_func,
            unflatten_func,
            path_entry_type,
            namespace,
        )?;
        Ok(())
    }

    fn unregister_impl(
        &'static mut self,
        cls: &Bound<'_, PyType>,
        namespace: &str,
    ) -> PyResult<()> {
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
        PyTreeTypeRegistry::get_singleton(cls.py(), false).unregister_impl(cls, namespace)?;
        PyTreeTypeRegistry::get_singleton(cls.py(), true).unregister_impl(cls, namespace)?;
        Ok(())
    }

    #[inline]
    pub fn is_dict_insertion_ordered(
        namespace: Option<&str>,
        inherit_global_namespace: Option<bool>,
    ) -> bool {
        let namespace = namespace.unwrap_or("");
        let inherit_global_namespace = inherit_global_namespace.unwrap_or(true);

        #[allow(static_mut_refs)]
        let dict_insertion_ordered_namespaces =
            unsafe { DICT_INSERTION_ORDERED_NAMESPACES.get_or_init(HashSet::new) };

        dict_insertion_ordered_namespaces.contains(namespace)
            || (inherit_global_namespace && dict_insertion_ordered_namespaces.contains(""))
    }

    #[inline]
    pub fn set_dict_insertion_ordered(mode: bool, namespace: Option<&str>) {
        let namespace = namespace.unwrap_or("");

        #[allow(static_mut_refs)]
        unsafe {
            DICT_INSERTION_ORDERED_NAMESPACES.get_or_init(HashSet::new);
        }

        #[allow(static_mut_refs)]
        let dict_insertion_ordered_namespaces =
            unsafe { DICT_INSERTION_ORDERED_NAMESPACES.get_mut() }.unwrap();

        if mode {
            dict_insertion_ordered_namespaces.insert(namespace.into());
        } else {
            dict_insertion_ordered_namespaces.remove(namespace);
        }
    }
}

impl Drop for PyTreeTypeRegistry {
    fn drop(&mut self) {
        Python::with_gil(|_py| {
            self.registrations.clear();
            self.named_registrations.clear();
            self.builtin_types.clear();
        })
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
