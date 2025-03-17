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

use crate::rustree::pytypes::{is_namedtuple_class, is_structseq_class};
use crate::rustree::registry::registry_lookup;

#[pyfunction]
#[pyo3(signature = (obj, /, namespace=""))]
#[inline]
pub fn is_leaf(obj: &Bound<PyAny>, namespace: Option<&str>) -> PyResult<bool> {
    let cls = obj.get_type();
    match registry_lookup(&cls, namespace) {
        Some(_) => return Ok(false),
        _ => (),
    };
    Ok(!(is_namedtuple_class(&cls)? || is_structseq_class(&cls)?))
}
