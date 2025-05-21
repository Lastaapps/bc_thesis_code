"""
This module defines NAC coloring related types
"""

from typing import Collection, Tuple
from pyrigi.data_type import Edge

NACColoring = Tuple[Collection[Edge], Collection[Edge]]
