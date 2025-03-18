# Copyright 2024-2025 Xuehai Pan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

PROJECT_NAME   = rustree
COPYRIGHT      = "Xuehai Pan. All Rights Reserved."
SHELL          = /bin/bash
.SHELLFLAGS    := -eu -o pipefail -c
PROJECT_PATH   = $(PROJECT_NAME)
SOURCE_FOLDERS = $(PROJECT_PATH) include src tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.py" -o -iname "*.pyi") setup.py
RUST_FILES     = $(shell find $(SOURCE_FOLDERS) -type f -iname "*.rs") build.rs
COMMIT_HASH    = $(shell git rev-parse HEAD)
COMMIT_HASH_SHORT = $(shell git rev-parse --short=7 HEAD)
GOPATH         ?= $(HOME)/go
GOBIN          ?= $(GOPATH)/bin
PATH           := $(PATH):$(GOBIN)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTEST         ?= $(PYTHON) -X dev -m pytest
PYTESTOPTS     ?=

.PHONY: default
default: install

.PHONY: install
install:
	$(PYTHON) -m pip install -v .

.PHONY: install-editable install-e
install-editable install-e:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel
	$(PYTHON) -m pip install --upgrade maturin
	$(PYTHON) -m pip install -v --no-build-isolation --editable .

.PHONY: uninstall
uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

.PHONY: build
build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	find $(PROJECT_PATH) -type f -name '*.so' -delete
	find $(PROJECT_PATH) -type f -name '*.pxd' -delete
	rm -rf *.egg-info .eggs
	$(PYTHON) -m build --verbose

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(1))
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install --upgrade $(2))

.PHONY: pre-commit-install
pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

.PHONY: pyfmt-install
pyfmt-install:
	$(call check_pip_install,ruff)

.PHONY: ruff-install
ruff-install:
	$(call check_pip_install,ruff)

.PHONY: pylint-install
pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])
	$(call check_pip_install,pyenchant)

.PHONY: mypy-install
mypy-install:
	$(call check_pip_install,mypy)

.PHONY: xdoctest-install
xdoctest-install:
	$(call check_pip_install,xdoctest)

.PHONY: docs-install
docs-install:
	$(call check_pip_install,doc8)
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-rtd-theme)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install_extra,sphinxcontrib-spelling,sphinxcontrib-spelling pyenchant)

.PHONY: pytest-install
pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

.PHONY: test-install
test-install: pytest-install
	$(PYTHON) -m pip install --requirement tests/requirements.txt

.PHONY: rustfmt-install
rustfmt-install:
	rustup component add rustfmt

.PHONY: clippy-install
clippy-install:
	rustup component add clippy


.PHONY: go-install
go-install:
	command -v go || sudo apt-get satisfy -y 'golang (>= 1.16)'

.PHONY: addlicense-install
addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

.PHONY: pytest test
pytest test: pytest-install
	$(PYTEST) --version
	cd tests && $(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_NAME)' && \
	$(PYTHON) -X dev -W 'always' -W 'error' -c 'import $(PROJECT_NAME)._C; print(f"GLIBCXX_USE_CXX11_ABI={$(PROJECT_NAME)._C.GLIBCXX_USE_CXX11_ABI}")' && \
	$(PYTEST) --verbose --color=yes --durations=10 --showlocals \
		--cov="$(PROJECT_NAME)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

# Python Linters

.PHONY: pre-commit
pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit --version
	$(PYTHON) -m pre_commit run --all-files

.PHONY: pyfmt ruff-format
pyfmt ruff-format: pyfmt-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff format --check . && \
	$(PYTHON) -m ruff check --select=I .

.PHONY: ruff
ruff: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check .

.PHONY: ruff-fix
ruff-fix: ruff-install
	$(PYTHON) -m ruff --version
	$(PYTHON) -m ruff check --fix --exit-non-zero-on-fix .

.PHONY: pylint
pylint: pylint-install
	$(PYTHON) -m pylint --version
	$(PYTHON) -m pylint $(PROJECT_PATH)

.PHONY: mypy
mypy: mypy-install
	$(PYTHON) -m mypy --version
	$(PYTHON) -m mypy .

.PHONY: xdoctest doctest
xdoctest doctest: xdoctest-install
	$(PYTHON) -m xdoctest --version
	$(PYTHON) -m xdoctest --global-exec "from $(PROJECT_NAME) import *" $(PROJECT_NAME)

# Rust Linters

.PHONY: clippy
clippy: clippy-install
	cargo clippy --version
	cargo clippy --lib --all-targets --all-features --package $(PROJECT_NAME)

.PHONY: clippy-fix
clippy-fix: clippy-install
	cargo clippy --version
	cargo clippy --lib --all-targets --all-features --allow-staged --allow-dirty --fix -p $(PROJECT_NAME)

.PHONY: rustfmt cargo-fmt
rustfmt cargo-fmt: rustfmt-install
	cargo fmt --version
	cargo fmt --package $(PROJECT_NAME)

# Documentation

.PHONY: addlicense
addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2024-$(shell date +"%Y") \
		-ignore tests/coverage.xml -check $(SOURCE_FOLDERS)

.PHONY: docstyle
docstyle: docs-install
	make -C docs clean || true
	$(PYTHON) -m doc8 docs && make -C docs html SPHINXOPTS="-W"

.PHONY: docs
docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

.PHONY: spelling
spelling: docs-install
	make -C docs clean || true
	make -C docs spelling SPHINXOPTS="-W"

.PHONY: clean-docs
clean-docs:
	make -C docs clean || true

# Utility Functions

.PHONY: lint
lint: pyfmt ruff pylint mypy doctest rustfmt clippy addlicense docstyle spelling

.PHONY: format
format: pyfmt-install ruff-install addlicense-install
	$(PYTHON) -m ruff format $(PYTHON_FILES)
	$(PYTHON) -m ruff check --fix --exit-zero .
	cargo fmt --package $(PROJECT_NAME)
	addlicense -c $(COPYRIGHT) -l apache -y 2024-$(shell date +"%Y") \
		-ignore tests/coverage.xml $(SOURCE_FOLDERS)

.PHONY: clean-python
clean-python:
	find . -type f -name '*.py[cod]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".ruff_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm -f tests/.coverage tests/.coverage.* tests/coverage.xml tests/coverage-*.xml tests/coverage.*.xml
	rm -f tests/.junit tests/.junit.* tests/junit.xml tests/junit-*.xml tests/junit.*.xml

.PHONY: clean-rust
clean-rust:
	cargo clean

.PHONY: clean-build
clean-build:
	rm -rf build/ dist/
	find $(PROJECT_PATH) -type f -name '*.so' -delete
	find $(PROJECT_PATH) -type f -name '*.pxd' -delete
	rm -rf *.egg-info .eggs

.PHONY: clean
clean: clean-python clean-rust clean-build clean-docs
