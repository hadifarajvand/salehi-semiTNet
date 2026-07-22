.PHONY: preflight smoke reference bootstrap setup-full download prepare inference full compare verify package test

preflight:
	python scripts/preflight.py

smoke:
	python scripts/run_smoke.py
	python scripts/acceptance_check.py --mode smoke

reference:
	python scripts/generate_reference_outputs.py

bootstrap:
	python scripts/bootstrap_upstream.py

setup-full:
	bash scripts/setup_full_env.sh

download:
	python scripts/download_assets.py --all

prepare:
	python scripts/inspect_dataset.py --root data/raw/TSI15k
	python scripts/prepare_dataset.py --source data/raw/TSI15k --dest data/processed/TSI15k

inference:
	bash run_inference.sh

full:
	bash run_full.sh

compare:
	python scripts/compare_to_paper.py --results outputs/inference/metrics.json

verify:
	python scripts/acceptance_check.py --mode inference

test:
	python -m unittest discover -s tests -v

package:
	python scripts/package_deliverable.py
