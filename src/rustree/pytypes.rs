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
use pyo3::types::*;

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
    Ok(cls.is_instance_of::<PyType>() && is_namedtuple_class_impl(cls.downcast::<PyType>()?))
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_namedtuple_instance(obj: &Bound<PyAny>) -> PyResult<bool> {
    Ok(!obj.is_instance_of::<PyType>() && is_namedtuple_class_impl(&obj.get_type()))
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
    if !is_namedtuple_class_impl(cls) {
        let err_msg = format!("{}, got {}.", err_msg, obj.repr()?);
        return Err(PyTypeError::new_err(err_msg));
    }
    cls.getattr("_fields")?.extract()
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_namedtuple(obj: &Bound<PyAny>) -> PyResult<bool> {
    let cls = if obj.is_instance_of::<PyType>() {
        &obj.get_type()
    } else {
        obj.downcast::<PyType>()?
    };
    Ok(is_namedtuple_class_impl(cls))
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
    Ok(cls.is_instance_of::<PyType>() && is_structseq_class_impl(cls.downcast::<PyType>()?))
}

#[pyfunction]
#[pyo3(signature = (obj, /))]
#[inline]
pub fn is_structseq_instance(obj: &Bound<PyAny>) -> PyResult<bool> {
    Ok(!obj.is_instance_of::<PyType>() && is_structseq_class_impl(&obj.get_type()))
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
    Ok(is_structseq_class_impl(cls))
}

#[inline]
fn structseq_fields_impl<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyTuple>> {
    let py = cls.py();
    let fields = PyList::empty(py);

    #[cfg(PyPy)]
    {
        let locals = PyDict::new(py);
        locals.set_item("cls", cls)?;
        locals.set_item("fields", fields)?;
        py.run(
            ffi::c_str!(
                r#"
            from _structseq import structseqfield

            indices_by_name = {
                name: member.index
                for name, member in vars(cls).items()
                if isinstance(member, structseqfield)
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
        let n_sequence_fields = cls.getattr("n_sequence_fields")?.extract::<isize>()?;
        let members = unsafe { (*cls.as_type_ptr()).tp_members };
        // Fill tuple with member names
        for i in 0..n_sequence_fields {
            let member = unsafe { &*members.offset(i) };
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
    if !is_structseq_class_impl(cls) {
        let err_msg = format!("{}, got {}.", err_msg, obj.repr()?);
        return Err(PyTypeError::new_err(err_msg));
    }
    structseq_fields_impl(cls)
}
