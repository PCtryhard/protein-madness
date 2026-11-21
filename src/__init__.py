"""Motif-anchored protein design with score-guided search.

One forward pass: seed PDB with fixed motif residues -> RFdiffusion partial
diffusion -> LigandMPNN -> Boltz-2 -> hydrogen placement -> scoring. A search
loop then expands the best designs through two branches and re-scores them
until the score stops improving.

See PIPELINE.md for the specification this implements.
"""

__all__ = ["config", "runner", "scoring", "search", "stages", "store", "tracking"]
