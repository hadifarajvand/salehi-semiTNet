.PHONY: install download smoke full reference test package

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
	.venv/bin/python scripts/generate_reference_outputs.py

test:
	python project.py install
	.venv/bin/python -m unittest discover -s tests -v

package:
	python project.py install
	.venv/bin/python scripts/package_deliverable.py
