# trajopt-jax

Trajectory optimisation in JAX, built for the GPU: an interior-point solver with Riccati linear algebra, and a Radau
collocation transcription for vehicle models (events, chained missions, co-design, mesh refinement).

```bash
pip install git+https://github.com/thearn/trajopt-jax
```

Example: [`examples/brachistochrone.py`](examples/brachistochrone.py), and as a notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thearn/trajopt-jax/blob/main/examples/brachistochrone.ipynb)
