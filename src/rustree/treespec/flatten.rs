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
use crate::rustree::registry::PyTreeTypeRegistry;

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn is_leaf(
    obj: &Bound<PyAny>,
    leaf_predicate: Option<&Bound<PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<bool> {
    let cls = obj.get_type();
    if leaf_predicate.is_some() {
        let result = leaf_predicate.unwrap().call1((obj,))?;
        if result.is_truthy()? {
            return Ok(true);
        }
    }
    if PyTreeTypeRegistry::lookup(&cls, none_is_leaf, namespace).is_some() {
        return Ok(false);
    };
    Ok(!(is_namedtuple_class(&cls)? || is_structseq_class(&cls)?))
}
