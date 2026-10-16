# The snapshot's tests need Parcae's `parcae_lm` package and a GPU; they are a reference copy,
# not part of MORPH's suite. Keep a bare `pytest` from the repo root from collecting them.
collect_ignore_glob = ["*"]
