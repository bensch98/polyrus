<h1 align="center">
<img src="https://raw.githubusercontent.com/bensch98/polyrus/main/docs/source/_static/img/polyrus_logo.png" width="300">
</h1><br>

# `polyrus`

## Polyrus: Utility library for geometric deep learning

`polyrus` is a Python library that implements multiple functionalities regarding 3-dimensional data structures.
As of now it's meant for:
- 3D data preprocessing for deep learning on manifolds
- visualizing results
- implementing new algorithms based on commonly used ones
- computing common metrics

## Installation

Install the latest Polyrus version with:
```bash
pip install polyrus
```

Install Polyrus with all optional dependencies.
```bash
pip install 'polyrus[all]'
```

Install Polyrus with a subset of all optional dependencies.
```bash
pip install 'polyrus[dev]'
```

To see the current Polyrus version in use, run:
```bash
import polyrus
polyrus.__version__
```
