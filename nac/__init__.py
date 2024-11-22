"""
Module related to the NAC coloring search
"""

from nac.data_type import *
from nac.monochromatic_classes import (
    find_triangle_components,
    fake_triangle_components,
    create_T_graph_from_components,
)
from nac.entry import *
from nac.util import canonical_NAC_coloring

from nac.search import (
    find_cycles_in_T_graph,
    find_useful_cycles_for_components,
    find_useful_cycles,
    find_shortest_cycles_for_components,
    find_shortest_cycles,
)
from nac.existence import (
    _check_for_simple_stable_cut,
)

from nac.check import (
    is_NAC_coloring,
    is_cartesian_NAC_coloring,
    NAC_check_called,
)
