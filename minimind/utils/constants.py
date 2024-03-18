import os

import jax

import minimind

# ZIPFILE_PATH_GCS = "gs://minimind/copies"
ZIPFILE_PATH_GCS = "/tmp/minimind/copies"

MINIMIND_ROOT_DIR = os.path.dirname(minimind.__path__[0])
MINIMIND_PATH = minimind.__path__[0]


## Jax
JAX_DEFAULT_BACKEND = jax.default_backend()
JAX_DEFAULT_PLATFORM = jax.lib.xla_bridge.get_backend().platform
