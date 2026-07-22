# Third-party notices

This project implements the experiment workflow around the method described in the following publication:

> A Semi-Supervised Transformer-Based Deep Learning Framework for Automated Tooth Segmentation and Identification on Panoramic Radiographs, Diagnostics 2024, DOI `10.3390/diagnostics14171948`.

The full-model execution path uses public research code and software components that are downloaded as dependencies during `make bootstrap`. The dependency is pinned to a specific public commit for stable execution and is kept under `vendor/` after setup.

The project-specific code in this repository covers dataset download/preparation, configuration, smoke simulation, training orchestration, evaluation, metric calculation, visualization, result comparison, environment setup and packaging.

Third-party components retain their original licenses and copyright notices. Users should review the license files included with downloaded dependencies before redistribution or commercial use.

The TSI15k dataset and released SemiTNet checkpoint are downloaded from their public hosting locations by `scripts/download_assets.py`; they are not redistributed directly in this repository.