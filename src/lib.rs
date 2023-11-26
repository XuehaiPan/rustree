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

mod rustree;

#[pymodule]
#[pyo3(name = "_rs")]
fn build_extension(m: &Bound<PyModule>) -> PyResult<()> {
    m.add("Py_TPFLAGS_BASETYPE", ffi::Py_TPFLAGS_BASETYPE)?;
    m.add_class::<rustree::PyTreeKind>()?;
    m.add_function(wrap_pyfunction!(rustree::is_namedtuple, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::is_namedtuple_instance, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::is_namedtuple_class, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::namedtuple_fields, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::is_structseq, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::is_structseq_instance, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::is_structseq_class, m)?)?;
    m.add_function(wrap_pyfunction!(rustree::structseq_fields, m)?)?;
    Ok(())
}
