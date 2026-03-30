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

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

mod pytypes;
mod registry;
mod treespec;

#[pymodule(gil_used = false)]
#[pyo3(name = "_rs")]
fn build_extension(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("Py_TPFLAGS_BASETYPE", ffi::Py_TPFLAGS_BASETYPE)?;
    m.add(
        "__doc__",
        "Optimized PyTree Utilities. (C extension module built from src/rustree.cpp)",
    )?;
    m.add("Py_DEBUG", cfg!(py_sys_config = "Py_DEBUG"))?;
    m.add("Py_GIL_DISABLED", cfg!(Py_GIL_DISABLED))?;
    m.add("RUSTREE_HAS_NATIVE_ENUM", true)?;
    m.add("RUSTREE_HAS_SUBINTERPRETER_SUPPORT", false)?;
    m.add("RUSTREE_HAS_READ_WRITE_LOCK", true)?;
    crate::registry::add_pytree_kind_enum(m)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_namedtuple, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_namedtuple_instance, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_namedtuple_class, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::namedtuple_fields, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_structseq, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_structseq_instance, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::is_structseq_class, m)?)?;
    m.add_function(wrap_pyfunction!(crate::pytypes::structseq_fields, m)?)?;
    m.add_function(wrap_pyfunction!(crate::registry::register_node, m)?)?;
    m.add_function(wrap_pyfunction!(crate::registry::unregister_node, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::registry::is_dict_insertion_ordered,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::registry::set_dict_insertion_ordered,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::registry::get_registry_size, m)?)?;

    m.add("MAX_RECURSION_DEPTH", crate::treespec::MAX_RECURSION_DEPTH)?;
    m.add_class::<crate::treespec::PyTreeSpec>()?;
    m.add_class::<crate::treespec::PyTreeIter>()?;
    m.add_function(wrap_pyfunction!(crate::treespec::is_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::flatten, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::flatten_with_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::_deserialize_treespec, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::pytreespec_apply, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::make_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::make_none, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::make_from_collection, m)?)?;

    let globals = PyDict::new(m.py());
    globals.set_item("_C", m)?;
    m.py().run(
        ffi::c_str!(
            r#"
def _rustree_bind_traverse(cls, func):
    def traverse(self, leaves, /, f_node=None, f_leaf=None):
        return func(self, leaves, f_node, f_leaf)
    cls.traverse = traverse
_rustree_bind_traverse(_C.PyTreeSpec, _C._pytreespec_traverse)
del _rustree_bind_traverse
"#
        ),
        Some(&globals),
        None,
    )?;

    crate::pytypes::get_rust_module(m.py(), Some(m.clone().unbind()));
    Ok(())
}
