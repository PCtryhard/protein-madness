"""Directed evolution against our own copy of the judges' scoring function.

The loop: backbone -> sequence -> fold with the transition-state analogue ->
score -> mutate the good ones -> fold again. Every generation is a real
measurement through :mod:`src.score.official`, never a surrogate, because the
thing we are scored on is what Boltz-2 does to our sequence.

Modules:

* :mod:`~src.evolve.mutate`    - the mutation operator (what changes, and how much)
* :mod:`~src.evolve.select`    - fitness and who becomes a parent
* :mod:`~src.evolve.diversity` - the guards against collapsing onto one design
* :mod:`~src.evolve.fold_batch`- Boltz input/output for a whole generation
* :mod:`~src.evolve.driver`    - generations, state on disk, resumability

See BRIEF.md 4 for why each piece is shaped the way it is.
"""
