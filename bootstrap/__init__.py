"""Setup-time modules: clone, patch, resolve models, build environments.

This package is deliberately NOT called `setup`. Kaggle's Python image ships an
installed distribution that owns that name (`dist-packages/setup/__init__.py`),
and a directory without `__init__.py` is only a *namespace* package -- which
loses to any real installed package of the same name no matter how early its
parent sits on sys.path. The result was:

    ImportError: cannot import name 'prepare_env' from 'setup'
                 (/usr/local/lib/python3.12/dist-packages/setup/__init__.py)

Hence both defences: a name nobody else claims, and a real `__init__.py` so
sys.path order is what decides.
"""
