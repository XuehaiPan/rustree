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

use std::sync::Arc;

use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::*;

use crate::rustree::pytypes::{is_namedtuple_class, is_structseq_class};
use crate::rustree::registry::{PyTreeKind, PyTreeTypeRegistration, PyTreeTypeRegistry};

use crate::rustree::treespec::PyTreeSpec;
use crate::rustree::treespec::treespec::Node;

pub const MAX_RECURSION_DEPTH: usize = 1000;

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn is_leaf(
    obj: &Bound<PyAny>,
    leaf_predicate: Option<&Bound<PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<bool> {
    if leaf_predicate.is_some() && leaf_predicate.unwrap().call1((obj,))?.is_truthy()? {
        return Ok(true);
    }
    if let (PyTreeKind::Leaf, ..) = PyTreeTypeRegistry::lookup(obj, none_is_leaf, namespace) {
        return Ok(true);
    }
    let cls = &obj.get_type();
    Ok(!(is_namedtuple_class(cls)? || is_structseq_class(cls)?))
}

impl PyTreeSpec {
    fn flatten_into_impl(
        obj: &Bound<PyAny>,
        traversal: &mut Vec<Node>,
        leaves: &mut Vec<Py<PyAny>>,
        depth: usize,
        leaf_predicate: Option<&Bound<PyAny>>,
        none_is_leaf: bool,
        namespace: &str,
    ) -> PyResult<bool> {
        if depth > MAX_RECURSION_DEPTH {
            return Err(PyRecursionError::new_err(
                "Maximum recursion depth exceeded during flattening the tree.",
            ));
        }
        let mut found_custom = false;
        let start_num_nodes = traversal.len();
        let start_num_leaves = leaves.len();

        let mut node = Node::default();

        if leaf_predicate.is_some() && leaf_predicate.unwrap().call1((obj,))?.is_truthy()? {
            leaves.push(obj.clone().unbind());
        } else {
            let registration: Option<Arc<PyTreeTypeRegistration>>;
            (node.kind, registration) =
                PyTreeTypeRegistry::lookup(obj, Some(none_is_leaf), Some(namespace));

            let mut recurse = |child| {
                Self::flatten_into_impl(
                    &child,
                    traversal,
                    leaves,
                    depth + 1,
                    leaf_predicate,
                    none_is_leaf,
                    namespace,
                )
            };

            match node.kind {
                PyTreeKind::Leaf => {
                    leaves.push(obj.clone().unbind());
                }
                PyTreeKind::None => {
                    if none_is_leaf {
                        unreachable!("None should be a leaf");
                    }
                }
                PyTreeKind::Tuple => {
                    let obj = obj.downcast::<PyTuple>()?;
                    node.arity = obj.len();
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::List => {
                    let obj = obj.downcast::<PyList>()?;
                    node.arity = obj.len();
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::Dict | PyTreeKind::OrderedDict | PyTreeKind::DefaultDict => {
                    let obj = obj.downcast::<PyDict>()?;
                    node.arity = obj.len();
                    let keys = obj.keys();
                    node.original_keys = Some(keys.call_method0("copy")?.unbind());
                    if node.kind != PyTreeKind::OrderedDict {
                        keys.sort()?;
                    }
                    for key in &keys {
                        let child = obj.get_item(key)?.unwrap();
                        found_custom |= recurse(child.clone())?;
                    }
                    if node.kind == PyTreeKind::DefaultDict {
                        let default_factory = obj.getattr("default_factory")?;
                        node.node_data = Some(
                            PyTuple::new(obj.py(), &[default_factory, keys.into_any()])?
                                .unbind()
                                .into_any(),
                        );
                    } else {
                        node.node_data = Some(keys.unbind().into_any());
                    }
                }
                PyTreeKind::NamedTuple | PyTreeKind::StructSequence => {
                    let obj = obj.downcast::<PyTuple>()?;
                    node.arity = obj.len();
                    node.node_data = Some(obj.get_type().unbind().into_any());
                    for child in obj {
                        found_custom |= recurse(child)?;
                    }
                }
                PyTreeKind::Deque => {
                    let list =
                        unsafe { obj.clone().downcast_into_unchecked::<PySequence>() }.to_list()?;
                    node.arity = list.len();
                    for child in list {
                        found_custom |= recurse(child)?;
                    }
                    node.node_data = Some(obj.getattr("maxlen")?.unbind());
                }
                PyTreeKind::Custom => {
                    found_custom = true;
                    let flatten_func = registration
                        .unwrap()
                        .flatten_func
                        .as_ref()
                        .unwrap()
                        .bind(obj.py())
                        .clone();
                    let out = flatten_func.call1((obj,))?;
                    let out = unsafe { out.downcast_into_unchecked::<PySequence>() }.to_tuple()?;
                    if out.len() != 2 && out.len() != 3 {
                        return Err(PyValueError::new_err(
                            "Custom flatten function must return a tuple of length 2 or 3.",
                        ));
                    }
                    node.node_data = Some(out.get_item(1)?.unbind());
                    let children = out.get_item(0)?;
                    for child in children.try_iter()? {
                        found_custom |= recurse(child?)?;
                        node.arity += 1;
                    }
                    if out.len() == 3 {
                        let node_entries = out.get_item(2)?;
                        if !node_entries.is_none() {
                            let node_entries =
                                unsafe { node_entries.downcast_into_unchecked::<PySequence>() }
                                    .to_tuple()?;
                            if node_entries.len() != node.arity {
                                return Err(PyValueError::new_err(
                                    "Custom flatten function must return a tuple of length 3, where the third element is a tuple of the same length as the first element.",
                                ));
                            }
                            node.node_entries = Some(node_entries.unbind());
                        }
                    }
                }
            }
        }

        node.num_leaves = leaves.len() - start_num_leaves;
        node.num_nodes = traversal.len() - start_num_nodes + 1;
        traversal.push(node);
        Ok(found_custom)
    }

    pub fn flatten(
        obj: &Bound<PyAny>,
        leaf_predicate: Option<&Bound<PyAny>>,
        none_is_leaf: bool,
        namespace: &str,
    ) -> PyResult<(Vec<Py<PyAny>>, PyTreeSpec)> {
        let mut traversal = Vec::new();
        let mut leaves = Vec::new();
        let found_custom = Self::flatten_into_impl(
            obj,
            &mut traversal,
            &mut leaves,
            0,
            leaf_predicate,
            none_is_leaf,
            namespace,
        )?;
        let namespace = if found_custom {
            String::from(namespace)
        } else {
            String::from("")
        };

        let treespec = PyTreeSpec::new(traversal, none_is_leaf, namespace);
        Ok((leaves, treespec))
    }
}

#[pyfunction]
#[pyo3(signature = (obj, /, leaf_predicate=None, none_is_leaf=false, namespace=""))]
#[inline]
pub fn flatten(
    obj: &Bound<'_, PyAny>,
    leaf_predicate: Option<&Bound<'_, PyAny>>,
    none_is_leaf: Option<bool>,
    namespace: Option<&str>,
) -> PyResult<(Vec<Py<PyAny>>, PyTreeSpec)> {
    let none_is_leaf = none_is_leaf.unwrap_or(false);
    let namespace = namespace.unwrap_or("");
    PyTreeSpec::flatten(obj, leaf_predicate, none_is_leaf, namespace)
}
