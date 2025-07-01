## Vortex Mamba for Hyperspectral Image Classification

## Abstract

Accurate classification of hyperspectral imagery (HSI) is critical for remote sensing analysis. While Transformers excel at modeling long-range dependencies in HSI, their quadratic complexity imposes prohibitive computational burdens. Mamba offers linear efficiency but suffers from unidirectional scanning, inadequately modeling local spatial relationships. To overcome these limitations, we propose Vortex Mamba—a hybrid architecture integrating Mamba, U-Net, and Transformers for complementary spectro-spatial feature extraction. Our Vortex Mamba Module replaces unidirectional scanning with centripetal spiral traversal, propagating features from the spatial center outward to establish radial hierarchical dependencies and angular continuity. This enables direction-agnostic global modeling with linear complexity. Further, our Channel Enhanced Attention Module performs dynamic spectral recalibration, using the central pixel's feature vector to weight channels and refine semantics. Tests conducted on widely adopted hyperspectral datasets such as Indian Pines, Pavia University, and Houston 2013 show that Vortex Mamba outperforms leading modern models, reaching notable accuracy rates of 98.78%, 99.51%, and 99.61% respectively.

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

