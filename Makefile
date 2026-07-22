.PHONY: install download smoke full reference test package

install:
	bash scripts/install.sh

download:
	python scripts/download_assets.py --all

smoke:
	python scripts/run_smoke.py
	python scripts/acceptance_check.py --mode smoke

full:
	bash scripts/run_full_easy.sh

reference:
	python scripts/generate_reference_outputs.py

test:
	python -m unittest discover -s tests -v

package:
	python scripts/package_deliverable.py
