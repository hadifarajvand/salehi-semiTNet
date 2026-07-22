.PHONY: install download smoke full reference test package

PY := .venv/bin/python

install:
	bash scripts/install.sh

download:
	$(PY) scripts/download_assets.py --all

smoke:
	$(PY) scripts/run_smoke.py
	$(PY) scripts/acceptance_check.py --mode smoke

full:
	bash scripts/run_full_easy.sh

reference:
	$(PY) scripts/generate_reference_outputs.py

test:
	$(PY) -m unittest discover -s tests -v

package:
	$(PY) scripts/package_deliverable.py
