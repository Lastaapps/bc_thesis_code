"""
This module defines NAC coloring related types
"""

from typing import Collection, Tuple

Edge = Tuple[int, int]
NACColoring = Tuple[Collection[Edge], Collection[Edge]]
