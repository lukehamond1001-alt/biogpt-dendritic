"""
BioGPT -- Biologically-Inspired Dendritic Language Model

A novel transformer architecture that replaces standard attention layers
with biologically-motivated dendritic neurons and cortical columns
inspired by mammalian neocortex.
"""

from biogpt.model import BioGPT, create_biogpt, count_parameters

__version__ = "0.1.0"
__all__ = ["BioGPT", "create_biogpt", "count_parameters"]
