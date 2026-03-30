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

use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::*;
use std::sync::{OnceLock, RwLock};

pub fn get_rust_module(py: Python<'_>, module: Option<Py<PyModule>>) -> &Py<PyModule> {
    static RUST_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    RUST_MODULE.get_or_init(py, || {
        assert!(module.is_some());
        module.unwrap()
    })
}

pub fn get_ordereddict(py: Python) -> Py<PyType> {
    static ORDEREDDICT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    ORDEREDDICT
        .import(py, "collections", "OrderedDict")
        .unwrap()
        .extract::<Bound<PyType>>()
        .unwrap()
        .unbind()
}

pub fn get_defaultdict(py: Python) -> Py<PyType> {
    static DEFAULTDICT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    DEFAULTDICT
        .import(py, "collections", "defaultdict")
        .unwrap()
        .extract::<Bound<PyType>>()
        .unwrap()
        .unbind()
}

pub fn get_deque(py: Python) -> Py<PyType> {
    static DEQUE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    DEQUE
        .import(py, "collections", "deque")
        .unwrap()
        .extract::<Bound<PyType>>()
        .unwrap()
        .unbind()
}

fn get_weak_key_dictionary<'py>(
    py: Python<'py>,
    cache: &'py PyOnceLock<Py<PyAny>>,
) -> PyResult<&'py Py<PyAny>> {
    cache.get_or_try_init(py, || {
        Ok(py
            .import("weakref")?
            .getattr("WeakKeyDictionary")?
            .call0()?
            .unbind())
    })
}

fn get_cache_lock(lock: &OnceLock<RwLock<()>>) -> &RwLock<()> {
    lock.get_or_init(|| RwLock::new(()))
}

fn get_cached_bool(
    py: Python<'_>,
    cache: &PyOnceLock<Py<PyAny>>,
    lock: &OnceLock<RwLock<()>>,
    cls: &Bound<'_, PyType>,
) -> PyResult<Option<bool>> {
    let _guard = get_cache_lock(lock).read().unwrap();
    let value = get_weak_key_dictionary(py, cache)?
        .bind(py)
        .call_method1("get", (cls,))?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract()?))
    }
}

fn set_cached_bool(
    py: Python<'_>,
    cache: &PyOnceLock<Py<PyAny>>,
    lock: &OnceLock<RwLock<()>>,
    cls: &Bound<'_, PyType>,
    value: bool,
) -> PyResult<bool> {
    let _guard = get_cache_lock(lock).write().unwrap();
    let cache = get_weak_key_dictionary(py, cache)?.bind(py);
    let cached = cache.call_method1("get", (cls,))?;
    if cached.is_none() {
        cache.set_item(cls, value)?;
        Ok(value)
    } else {
        cached.extract()
    }
}

fn get_cached_tuple<'py>(
    py: Python<'py>,
    cache: &PyOnceLock<Py<PyAny>>,
    lock: &OnceLock<RwLock<()>>,
    cls: &Bound<'py, PyType>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let _guard = get_cache_lock(lock).read().unwrap();
    let value = get_weak_key_dictionary(py, cache)?
        .bind(py)
        .call_method1("get", (cls,))?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.cast_into::<PyTuple>()?))
    }
}

fn set_cached_tuple<'py>(
    py: Python<'py>,
    cache: &PyOnceLock<Py<PyAny>>,
    lock: &OnceLock<RwLock<()>>,
    cls: &Bound<'py, PyType>,
    value: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let _guard = get_cache_lock(lock).write().unwrap();
    let cache = get_weak_key_dictionary(py, cache)?.bind(py);
    let cached = cache.call_method1("get", (cls,))?;
    if cached.is_none() {
        cache.set_item(cls, &value)?;
        Ok(value)
    } else {
        Ok(cached.cast_into::<PyTuple>()?)
    }
}

fn is_namedtuple_class_cached(cls: &Bound<'_, PyType>) -> PyResult<bool> {
    static CACHE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();

    let py = cls.py();
    if let Some(value) = get_cached_bool(py, &CACHE, &LOCK, cls)? {
        return Ok(value);
    }
    set_cached_bool(py, &CACHE, &LOCK, cls, is_namedtuple_class_impl(cls))
}

fn is_structseq_class_cached(cls: &Bound<'_, PyType>) -> PyResult<bool> {
    static CACHE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();

    let py = cls.py();
    if let Some(value) = get_cached_bool(py, &CACHE, &LOCK, cls)? {
        return Ok(value);
    }
    set_cached_bool(py, &CACHE, &LOCK, cls, is_structseq_class_impl(cls))
}

fn structseq_fields_cached<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyTuple>> {
    static CACHE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static LOCK: OnceLock<RwLock<()>> = OnceLock::new();

    let py = cls.py();
    if let Some(fields) = get_cached_tuple(py, &CACHE, &LOCK, cls)? {
        return Ok(fields);
    }
    let fields = structseq_fields_impl(cls)?;
    set_cached_tuple(py, &CACHE, &LOCK, cls, fields)
}

#[inline]
fn is_namedtuple_class_impl(cls: &Bound<PyType>) -> bool {
    // We can only identify namedtuples heuristically, here by the presence of a _fields attribute.
    if unsafe {
        ffi::PyType_FastSubclass(
            cls.as_ptr() as *mut ffi::PyTypeObject,
            ffi::Py_TPFLAGS_TUPLE_SUBCLASS,
        ) != 0
    } {
        let fields = match cls.getattr("_fields") {
            Ok(fields) => fields,
            Err(_) => {
                unsafe {
                    ffi::PyErr_Clear();
                }
                return false;
            }
        };
        if fields.is_instance_of::<PyTuple>()
            && fields
                .downcast::<PyTuple>()
                .unwrap()
                .iter()
                .all(|field| field.is_instance_of::<PyString>())
        {
            for name in ["_make", "_asdict"] {
                match cls.getattr(name) {
                    Ok(attr) => {
                        if !attr.is_callable() {
                            return false;
                        }
                    }
                    Err(_) => {
                        unsafe {
                            ffi::PyErr_Clear();
                        }
                        return false;
                    }
                }
            }
            return true;
        }
    }
    false
}

#[pyfunction]
#[pyo3(signature = (cls, /))]
#[inline]
pub fn is_namedtuple_class(cls: &Bound<PyAny>) -> PyResult<bool> {
    Ok(cls.is_instance_of::<PyType>() && is_namedtuple_class_cached(cls.downcast::<PyType>()?)?)
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_namedtuple_instance(obj: &Bound<PyAny>) -> PyResult<bool> {
    Ok(!obj.is_instance_of::<PyType>() && is_namedtuple_class_cached(&obj.get_type())?)
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn namedtuple_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let (cls, err_msg) = if obj.is_instance_of::<PyType>() {
        (
            obj.downcast::<PyType>()?,
            "Expected a collections.namedtuple type",
        )
    } else {
        (
            &obj.get_type(),
            "Expected an instance of collections.namedtuple type",
        )
    };
    if !is_namedtuple_class_cached(cls)? {
        let err_msg = format!("{}, got {}.", err_msg, obj.repr()?);
        return Err(PyTypeError::new_err(err_msg));
    }
    cls.getattr("_fields")?.cast_into().map_err(PyErr::from)
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_namedtuple(obj: &Bound<PyAny>) -> PyResult<bool> {
    let cls = if obj.is_instance_of::<PyType>() {
        obj.downcast::<PyType>()?
    } else {
        &obj.get_type()
    };
    is_namedtuple_class_cached(cls)
}

#[inline]
fn is_structseq_class_impl(cls: &Bound<PyType>) -> bool {
    let type_ptr: *mut ffi::PyTypeObject = cls.as_type_ptr();
    if unsafe {
        ffi::PyType_IsSubtype(type_ptr, std::ptr::addr_of_mut!(ffi::PyTuple_Type)) != 0
            && ffi::PyType_HasFeature(type_ptr, ffi::Py_TPFLAGS_BASETYPE) == 0
    } {
        let tp_bases: *mut ffi::PyObject = unsafe { (*type_ptr).tp_bases };
        if unsafe {
            ffi::PyTuple_CheckExact(tp_bases) != 0
                && ffi::PyTuple_Size(tp_bases) == 1
                && ffi::PyTuple_GetItem(tp_bases, 0)
                    == (std::ptr::addr_of_mut!(ffi::PyTuple_Type) as *mut ffi::PyObject)
        } {
            for name in ["n_fields", "n_sequence_fields", "n_unnamed_fields"] {
                match cls.getattr(name) {
                    Ok(attr) => {
                        if !attr.is_exact_instance_of::<PyInt>() {
                            return false;
                        }
                    }
                    Err(_) => return false,
                }
            }
            return true;
        }
    }
    false
}

#[pyfunction]
#[pyo3(signature = (cls, /))]
#[inline]
pub fn is_structseq_class(cls: &Bound<PyAny>) -> PyResult<bool> {
    Ok(cls.is_instance_of::<PyType>() && is_structseq_class_cached(cls.downcast::<PyType>()?)?)
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_structseq_instance(obj: &Bound<PyAny>) -> PyResult<bool> {
    Ok(!obj.is_instance_of::<PyType>() && is_structseq_class_cached(&obj.get_type())?)
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_structseq(obj: &Bound<PyAny>) -> PyResult<bool> {
    let cls = if obj.is_instance_of::<PyType>() {
        obj.downcast::<PyType>()?
    } else {
        &obj.get_type()
    };
    is_structseq_class_cached(cls)
}

#[inline]
fn structseq_fields_impl<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyTuple>> {
    let py = cls.py();
    let fields = PyList::empty(py);

    #[cfg(PyPy)]
    {
        let locals = PyDict::new(py);
        locals.set_item("cls", cls)?;
        locals.set_item("fields", &fields)?;
        py.run(
            ffi::c_str!(
                r#"
                import sys

                StructSequenceFieldType = type(type(sys.version_info).major)
                indices_by_name = {
                    name: member.index
                    for name, member in vars(cls).items()
                    if isinstance(member, StructSequenceFieldType)
                }
                fields.extend(sorted(indices_by_name, key=indices_by_name.get)[:cls.n_sequence_fields])
                "#
            ),
            None,
            Some(&locals),
        )?;
    }

    #[cfg(not(PyPy))]
    {
        let n_sequence_fields = cls.getattr("n_sequence_fields")?.extract::<usize>()?;
        let members = unsafe { (*cls.as_type_ptr()).tp_members };
        // Fill tuple with member names
        for i in 0..n_sequence_fields {
            let member = unsafe { &*members.add(i) };
            let field = unsafe {
                std::ffi::CStr::from_ptr(member.name)
                    .to_string_lossy()
                    .into_owned()
            };
            fields.append(field)?;
        }
    }

    Ok(fields.to_tuple())
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn structseq_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let (cls, err_msg) = if obj.is_instance_of::<PyType>() {
        (
            obj.downcast::<PyType>()?,
            "Expected a PyStructSequence type",
        )
    } else {
        (
            &obj.get_type(),
            "Expected an instance of PyStructSequence type",
        )
    };
    if !is_structseq_class_cached(cls)? {
        let err_msg = format!("{}, got {}.", err_msg, obj.repr()?);
        return Err(PyTypeError::new_err(err_msg));
    }
    structseq_fields_cached(cls)
}
