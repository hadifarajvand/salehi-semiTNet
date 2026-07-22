.PHONY: install download smoke full reference test package

ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

install:
	python project.py install

download:
	python project.py download

smoke:
	python project.py smoke

full:
	python project.py full

reference:
	python project.py install
	$(VENV_PY) scripts/generate_reference_outputs.py

test:
	python project.py install
	$(VENV_PY) -m unittest discover -s tests -v

package:
	python project.py install
	$(VENV_PY) scripts/package_deliverable.py
