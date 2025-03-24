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

mod pytypes;
mod registry;
mod treespec;

#[pymodule]
#[pyo3(name = "_rs")]
fn build_extension(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("Py_TPFLAGS_BASETYPE", ffi::Py_TPFLAGS_BASETYPE)?;
    m.add_class::<crate::registry::PyTreeKind>()?;
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

    m.add("MAX_RECURSION_DEPTH", crate::treespec::MAX_RECURSION_DEPTH)?;
    m.add_class::<crate::treespec::PyTreeSpec>()?;
    m.add_function(wrap_pyfunction!(crate::treespec::is_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(crate::treespec::flatten, m)?)?;

    crate::pytypes::get_rust_module(m.py(), Some(m.clone().unbind()));
    Ok(())
}
