"""Run-time modules: service startup, the LLM server, the public tunnel.

Has an `__init__.py` on purpose. Without one this is merely a namespace package,
and any installed distribution named `runtime` in the notebook image would win
the import -- the same failure mode that the old `setup/` directory hit on
Kaggle. See `bootstrap/__init__.py` for the full story.
"""
