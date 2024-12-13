from typing import *
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import importlib
from random import Random
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline as backend_inline
from matplotlib.backends import backend_agg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import networkx as nx
import os
import time
import datetime
import signal
import itertools
import base64

from tqdm import tqdm

import nac as nac
import nac.util
from nac import MonochromaticClassType
from benchmarks import dataset

importlib.reload(nac)
importlib.reload(nac.util)
importlib.reload(dataset)


###############################################################################
# https://stackoverflow.com/a/75898999
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def copy_doc(wrapper: Callable[P, T]):
    """An implementation of functools.wraps."""

    def decorator(func: Callable) -> Callable[P, T]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator


@copy_doc(plt.figure)
def figure(num: Any = 1, *args, **kwargs) -> Figure:
    """Creates a figure that is independent on the global plt state"""
    fig = Figure(*args, **kwargs)

    def show():
        manager = backend_agg.new_figure_manager_given_figure(num, fig)
        display(
            manager.canvas.figure,
            metadata=backend_inline._fetch_figure_metadata(manager.canvas.figure),
        )
        manager.destroy()

    fig.show = show
    return fig


###############################################################################
class LazyList[T](List[T]):
    """
    Delays list creation until first request
    """

    def __init__(self, generator: Callable[[], Iterable[T]]):
        self._generator = generator
        self._list: List[T] | None = None

    def _get_list(self) -> List[T]:
        if self._list is None:
            self._list = list(self._generator())
        return self._list

    def __iter__(self) -> Iterator[T]:
        return self._get_list().__iter__()

    def __len__(self) -> int:
        return self._get_list().__len__()

    def __getitem__(self, i: SupportsIndex) -> T:
        return self._get_list().__getitem__(i)


###############################################################################
COLUMNS: List[str] = [
    "graph",
    "dataset",
    "vertex_no",
    "edge_no",
    "triangle_components_no",
    "monochromatic_classes_no",
    "relabel",
    "split",
    "merging",
    "subgraph_size",
    "used_monochromatic_classes",
    "nac_any_finished",
    "nac_first_coloring_no",
    "nac_first_mean_time",
    "nac_first_rounds",
    "nac_first_check_is_NAC",
    "nac_first_check_cycle_mask",
    "nac_all_coloring_no",
    "nac_all_mean_time",
    "nac_all_rounds",
    "nac_all_check_is_NAC",
    "nac_all_check_cycle_mask",
]


@dataclass
class MeasurementResult:
    graph: str
    dataset: str
    vertex_no: int
    edge_no: int
    triangle_components_no: int
    monochromatic_classes_no: int
    relabel: str
    split: str
    merging: str
    subgraph_size: int
    used_monochromatic_classes: bool
    nac_any_finished: bool
    nac_first_coloring_no: Optional[int]
    nac_first_mean_time: Optional[int]
    nac_first_rounds: Optional[int]
    nac_first_check_is_NAC: Optional[int]
    nac_first_check_cycle_mask: Optional[int]
    nac_all_coloring_no: Optional[int]
    nac_all_mean_time: Optional[int]
    nac_all_rounds: Optional[int]
    nac_all_check_is_NAC: Optional[int]
    nac_all_check_cycle_mask: Optional[int]

    def to_list(self) -> List:
        return [
            self.graph,
            self.dataset,
            self.vertex_no,
            self.edge_no,
            self.triangle_components_no,
            self.monochromatic_classes_no,
            self.relabel,
            self.split,
            self.merging,
            self.subgraph_size,
            self.used_monochromatic_classes,
            self.nac_any_finished,
            self.nac_first_coloring_no,
            self.nac_first_mean_time,
            self.nac_first_rounds,
            self.nac_first_check_is_NAC,
            self.nac_first_check_cycle_mask,
            self.nac_all_coloring_no,
            self.nac_all_mean_time,
            self.nac_all_rounds,
            self.nac_all_check_is_NAC,
            self.nac_all_check_cycle_mask,
        ]


###############################################################################
def toBenchmarkResults(data: List[MeasurementResult] = []) -> pd.DataFrame:
    return pd.DataFrame(
        [x.to_list() for x in data],
        columns=COLUMNS,
    )


def graph_to_id(graph: nx.Graph) -> str:
    return base64.standard_b64encode(
        nx.graph6.to_graph6_bytes(graph, header=False).strip()
    ).decode()


def graph_from_id(id: str) -> nx.Graph:
    return nac.util.NiceGraph(
        nx.graph6.from_graph6_bytes(base64.standard_b64decode(id))
    )


###############################################################################

OUTPUT_DIR: str = None
_BENCH_FILE_START = "bench_res"


def get_output_dir() -> str:
    return OUTPUT_DIR


def _find_latest_record_file(
    prefix: str,
    dir: str,
) -> str | None:
    def filter_cond(name: str) -> bool:
        return name.startswith(prefix) and name.endswith(".csv")

    data = sorted(filter(filter_cond, os.listdir(dir)), reverse=True)

    if len(data) == 0:
        return None
    file_name = data[0]
    return file_name


def load_records(
    file_name: str | None = None,
    dir: str | None = None,
    allow_output: bool = False,
) -> pd.DataFrame:
    """
    Loads the results from the last run or the run specified by `file_name` in the `dir` given.
    """
    dir = dir or OUTPUT_DIR
    if file_name == None:
        file_name = _find_latest_record_file(_BENCH_FILE_START, dir)
        if file_name is None:
            if allow_output:
                print(f"No file with results found in {dir}!")
            return toBenchmarkResults()
        print(f"Found file: {file_name}")

    path = os.path.join(dir, file_name)
    df = pd.read_csv(path)
    df = df[COLUMNS]
    assert len(df) > 0
    return df


def store_results(
    df: pd.DataFrame,
    file_name: str | None = None,
    dir: str | None = None,
) -> str:
    """
    Stores results in the given file
    """
    dir = dir or OUTPUT_DIR
    if file_name is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{_BENCH_FILE_START}_{current_time}.csv"
    path = os.path.join(dir, file_name)

    # Filter out outliers (over 60s) that run when I put my laptor into sleep mode
    df = df.query("nac_all_mean_time < 60_000 and nac_first_mean_time < 60_000")

    df.to_csv(path, header=True, index=False)
    return file_name


def update_stored_data(
    dfs: List[pd.DataFrame] = [], head_loaded: bool = True
) -> pd.DataFrame:
    df = load_records()
    if head_loaded:
        display(df)
    if len(dfs) != 0:
        df = pd.concat((df, pd.concat(dfs)))
    df = df.drop_duplicates(
        subset=[
            "graph",
            "dataset",
            "split",
            "relabel",
            "merging",
            "subgraph_size",
            "used_monochromatic_classes",
        ],
        keep="last",
    )
    store_results(df)
    return df


###############################################################################
class BenchmarkTimeoutException(Exception):
    def __init__(self, msg: str = "The benchmark timed out", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


def with_timeout[
    **T, R, D
](function: Callable[T, R], time_limit: int | None, default: D) -> Callable[T, R | D]:
    """
    Stops the function execution after a specified timeout in seconds is reached.
    """
    if time_limit is None:
        return function

    def impl(*args: P.args, **kwargs: P.kwargs):
        try:
            # signals are not exact, but generally work
            def timeout_handler(signum, frame):
                raise BenchmarkTimeoutException()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(time_limit)

            res = function(*args, **kwargs)

            signal.alarm(0)
            return res
        except BenchmarkTimeoutException:
            return default

    return impl


###############################################################################
@dataclass
class MeasuredRecord:
    """
    Pepresents measurement result of a single type
    """

    time_sum: int = 0
    coloring_no: int = 0
    rounds: int = 0
    checks_is_NAC: int = 0
    checks_cycle_mask: int = 0

    @property
    def mean_time(self) -> int:
        if self.rounds == 0:
            return 0
        return int(self.time_sum / self.rounds * 1000)


@dataclass
class MeasuredData:
    """
    Groups measurement results for both the types of tests
    """

    first: Optional[MeasuredRecord]
    all: Optional[MeasuredRecord]


###############################################################################


def nac_benchmark_core(
    graph: nx.Graph,
    rounds: int,
    first_only: bool,
    strategy: Tuple[str, str],
    use_monochromatic_classes: bool,
    time_limit: int,
    seed: int | None = 42,
) -> MeasuredData:
    """
    Runs benchmarks for NAC coloring search
    Returns results grouped by relabel, split, merge and subgraph size strategies
    """

    if use_monochromatic_classes:
        monochromatic_class_type = nac.MonochromaticClassType.MONOCHROMATIC
    else:
        monochromatic_class_type = nac.MonochromaticClassType.TRIANGLES

    result = MeasuredData(None, None)
    rand = random.Random(seed)

    def find_colorings():
        start_time = time.time()

        itr = iter(
            nac.NAC_colorings(
                graph=graph,
                algorithm=strategy[1],
                relabel_strategy=strategy[0],
                monochromatic_class_type=monochromatic_class_type,
                seed=rand.randint(0, 2**30),
            )
        )

        first_col = next(itr, None)
        first_time = time.time()

        if result.first is None:
            result.first = MeasuredRecord()
        result.first = MeasuredRecord(
            time_sum=result.first.time_sum + first_time - start_time,
            coloring_no=0 if first_col is None else 1,
            rounds=result.first.rounds + 1,
            checks_is_NAC=nac.NAC_check_called()[0],
            checks_cycle_mask=nac.NAC_check_called()[1],
        )

        if first_only:
            return

        j = 0
        for j, coloring in enumerate(itr):
            pass
        end_time = time.time()

        if result.all is None:
            result.all = MeasuredRecord()
        result.all = MeasuredRecord(
            time_sum=result.all.time_sum + end_time - start_time,
            coloring_no=j + 1 + 1,
            rounds=result.all.rounds + 1,
            checks_is_NAC=nac.NAC_check_called()[0],
            checks_cycle_mask=nac.NAC_check_called()[1],
        )

    def run() -> None:
        [find_colorings() for _ in range(rounds)]

    with_timeout(
        run,
        time_limit=time_limit * rounds,
        default=None,
    )()

    return result


###############################################################################
def create_measurement_result(
    graph: nx.Graph,
    dataset_name: str,
    trianlge_classes: int,
    monochromatic_classes: int,
    nac_first: Optional[MeasuredRecord],
    nac_all: Optional[MeasuredRecord],
    relabel_strategy: str,
    split_strategy: str,
    merge_strategy: str,
    subgraph_size: int,
    used_monochromatic_classes: bool,
) -> MeasurementResult:
    vertex_no = nx.number_of_nodes(graph)
    edge_no = nx.number_of_edges(graph)

    nac_any_finished = (nac_first or nac_all) is not None
    nac_first = nac_first or MeasuredRecord()
    nac_all = nac_all or MeasuredRecord()

    return MeasurementResult(
        graph=graph_to_id(graph),
        dataset=dataset_name,
        vertex_no=vertex_no,
        edge_no=edge_no,
        triangle_components_no=trianlge_classes,
        monochromatic_classes_no=monochromatic_classes,
        relabel=relabel_strategy,
        split=split_strategy,
        merging=merge_strategy,
        subgraph_size=subgraph_size,
        used_monochromatic_classes=used_monochromatic_classes,
        nac_any_finished=nac_any_finished,
        nac_first_coloring_no=nac_first.coloring_no,
        nac_first_mean_time=nac_first.mean_time,
        nac_first_rounds=nac_first.rounds,
        nac_first_check_is_NAC=nac_first.checks_is_NAC,
        nac_first_check_cycle_mask=nac_first.checks_cycle_mask,
        nac_all_coloring_no=nac_all.coloring_no,
        nac_all_mean_time=nac_all.mean_time,
        nac_all_rounds=nac_all.rounds,
        nac_all_check_is_NAC=nac_all.checks_is_NAC,
        nac_all_check_cycle_mask=nac_all.checks_cycle_mask,
    )


###############################################################################
###############################################################################
###############################################################################


def _group_and_plot(
    df: pd.DataFrame,
    axs: List[plt.Axes],
    x_column: Literal["vertex_no", "monochromatic_classes_no"],
    based_on: Literal["relabel", "split", "merging"],
    value_columns: List[Literal["nac_first_mean_time", "nac_all_mean_time"]],
):
    aggregations = ["mean", "median", "3rd quartile"]
    df = df.loc[:, [x_column, based_on, *value_columns]]
    groupped = df.groupby([x_column, based_on])

    for ax, aggregation in zip(axs, aggregations):
        match aggregation:
            case "mean":
                aggregated = groupped.mean()
            case "median":
                aggregated = groupped.median()
            case "3rd quartile":
                aggregated = groupped.quantile(0.75)

        aggregated = aggregated.reorder_levels([based_on, x_column], axis=0)

        for name in aggregated.index.get_level_values(based_on).unique():
            data = aggregated.loc[name]
            for value_column in value_columns:
                title = (
                    ",".join([name, value_column]) if len(value_columns) > 1 else name
                )
                ax.plot(data.index, data[value_column], label=title)

        rename_based_on = {
            "vertex_no": "Vertices",
            "triangle_components_no": "Triangle components",
            "monochromatic_classes_no": "Monochromatic classes",
        }

        # ax.set_title(f"{rename_based_on[x_column]} {based_on} ({aggregation})")
        # ax.set_title(f"{rename_based_on[x_column]} ({aggregation})")
        ax.set_title(f"{aggregation.capitalize()}")
        if "time" in value_columns[0]:
            ax.set_ylabel("Time [ms]")
        if "check" in value_columns[0]:
            ax.set_ylabel("Checks [call]")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(rename_based_on[x_column])
        ax.legend()


def plot_frame(
    title: str,
    df: pd.DataFrame,
    ops_value_columns_sets=[
        [
            "nac_first_mean_time",
        ],
        [
            "nac_first_check_cycle_mask",
        ],
        [
            "nac_all_mean_time",
        ],
        [
            "nac_all_check_cycle_mask",
        ],
    ],
    ops_x_column=[
        "vertex_no",
        "monochromatic_classes_no",
    ],
    ops_based_on=[
        #  "relabel",
        # "split",
        # "merging",
        "split_merging",
    ],
    ops_aggregation=[
        "mean",
        "median",
    ],  #  "3rd quartile",
) -> List[Figure]:
    print(f"Plotting {df.shape[0]} records...")
    figs = []

    title_rename = {
        "nac_first_mean_time": "First NAC-coloring, Runtime",
        "nac_first_check_cycle_mask": "First NAC-coloring, Checks number",
        "nac_all_mean_time": "All NAC-colorings, Runtime",
        "nac_all_check_cycle_mask": "All NAC-colorings, Checks number",
    }

    for value_columns in ops_value_columns_sets:
        local_df = df[(df[value_columns] != 0).all(axis=1)]
        if local_df.shape[0] == 0:
            continue

        nrows = len(ops_x_column) * len(ops_based_on)
        ncols = len(ops_aggregation)
        fig = figure(nrows * ncols, (20, 6 * nrows), layout="constrained")
        title_detail = " | ".join(
            title_rename[value_column] for value_column in value_columns
        )
        fig.suptitle(f"{title} ({title_detail})", fontsize=20)
        figs.append(fig)

        row = 0
        for x_column in ops_x_column:
            for based_on in ops_based_on:
                axs = [
                    fig.add_subplot(nrows, ncols, i + ncols * row + 1)
                    for i in range(len(ops_aggregation))
                ]
                _group_and_plot(local_df, axs, x_column, based_on, value_columns)
                row += 1
    return figs


###############################################################################
def _plot_is_NAC_coloring_calls_groups(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    x_column: Literal["vertex_no", "monochromatic_classes_no"],
    value_columns: List[Literal["nac_first_mean_time", "nac_all_mean_time"]],
    aggregation: Literal["mean", "median", "3rd quartile"],
    legend_rename_dict: Dict[str, str] = {},
):
    df = df.loc[:, [x_column, *value_columns]]
    groupped = df.groupby([x_column])
    match aggregation:
        case "mean":
            aggregated = groupped.mean()
        case "median":
            aggregated = groupped.median()
        case "3rd quartile":
            aggregated = groupped.quantile(0.75)

    rename_based_on = {
        "vertex_no": "Vertices",
        "triangle_components_no": "Triangle components",
        "monochromatic_classes_no": "Monochromatic classes",
    }

    # display(aggregated)
    aggregated.plot(ax=ax)
    ax.set_title(f"{title} - {aggregation.capitalize()}")
    ax.set_ylabel("Checks [call]")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(rename_based_on[x_column])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_rename_dict[l] for l in labels], loc="upper left")


def plot_is_NAC_coloring_calls(
    df: pd.DataFrame,
) -> List[Figure]:
    figs = []

    df = df.query("nac_all_coloring_no != 0").copy()
    print(f"Plotting {df.shape[0]} records...")

    related_columns = [
        "vertex_no",
        "edge_no",
        "triangle_components_no",
        "monochromatic_classes_no",
        "nac_all_coloring_no",
        "nac_all_check_is_NAC",
        "nac_all_check_cycle_mask",
    ]
    df = df.loc[:, related_columns]
    # this does not help our algorithm to stand out, but the graphs can be drawn more easily

    df["exp_edge_no"] = 2 ** (df["edge_no"] - 1)
    df["exp_triangle_component_no"] = 2 ** (df["triangle_components_no"] - 1)
    df["exp_monochromatic_class_no"] = 2 ** (df["monochromatic_classes_no"] - 1)

    df["scaled_edge_no"] = df["edge_no"] / df["nac_all_coloring_no"]
    df["scaled_triangle_component_no"] = (
        df["triangle_components_no"] / df["nac_all_coloring_no"]
    )
    df["scaled_monochromatic_class_no"] = (
        df["monochromatic_classes_no"] / df["nac_all_coloring_no"]
    )
    df["scaled_nac_all_check_cycle_mask"] = (
        df["nac_all_check_cycle_mask"] / df["nac_all_coloring_no"]
    )

    df["inv_edge_no"] = df["nac_all_coloring_no"] / df["edge_no"]
    df["inv_triangle_component_no"] = (
        df["nac_all_coloring_no"] / df["triangle_components_no"]
    )
    df["inv_monochromatic_class_no"] = (
        df["nac_all_coloring_no"] / df["monochromatic_classes_no"]
    )
    df["inv_nac_all_check_cycle_mask"] = (
        df["nac_all_coloring_no"] / df["nac_all_check_cycle_mask"]
    )
    df["inv_nac_all_check_is_NAC"] = (
        df["nac_all_coloring_no"] / df["nac_all_check_is_NAC"]
    )

    df["new_edge_no"] = df["edge_no"] / df["exp_monochromatic_class_no"]
    df["new_triangle_component_no"] = (
        df["triangle_components_no"] / df["exp_monochromatic_class_no"]
    )
    df["new_monochromatic_class_no"] = (
        df["monochromatic_classes_no"] / df["exp_monochromatic_class_no"]
    )
    df["new_nac_all_check_cycle_mask"] = (
        df["nac_all_check_cycle_mask"] / df["exp_monochromatic_class_no"]
    )

    rename_dict = {
        "exp_edge_no": "Naive - edges",
        "exp_triangle_component_no": "Naive - triangle-components",
        "exp_monochromatic_class_no": "Naive - monochromatic classes",
        "nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "nac_all_check_is_NAC": "Subgraphs - IsNACColoring",
        "scaled_edge_no": "Naive - edges",
        "scaled_triangle_component_no": "Naive - triangle-components",
        "scaled_monochromatic_class_no": "Naive - monochromatic classes",
        "scaled_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "inv_edge_no": "Naive - edges",
        "inv_triangle_component_no": "Naive - triangle-components",
        "inv_monochromatic_class_no": "Naive - monochromatic classes",
        "inv_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "inv_nac_all_check_is_NAC": "Subgraphs - IsNACColoring",
        "new_edge_no": "Naive - edges",
        "new_triangle_component_no": "Naive - triangle-components",
        "new_monochromatic_class_no": "Naive - monochromatic classes",
        "new_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
    }

    ops_x_column = [
        "vertex_no",
        "monochromatic_classes_no",
    ]
    ops_value_groups = [
        [
            "exp_edge_no",
            "exp_triangle_component_no",
            "exp_monochromatic_class_no",
            "nac_all_check_cycle_mask",
            "nac_all_check_is_NAC",
        ],
        # ["scaled_edge_no", "scaled_triangle_component_no", "scaled_monochromatic_class_no", "scaled_nac_all_check_cycle_mask"],
        [
            "inv_edge_no",
            "inv_triangle_component_no",
            "inv_monochromatic_class_no",
            "inv_nac_all_check_cycle_mask",
            "inv_nac_all_check_is_NAC",
        ],
        # ["new_edge_no",    "new_triangle_component_no",    "new_monochromatic_class_no",    "new_nac_all_check_cycle_mask" ],
    ]
    ops_aggregation = [
        "mean",
        "median",
    ]  # "3rd quartile",

    nrows = len(ops_value_groups)
    ncols = len(ops_aggregation)

    for x_column in ops_x_column:
        row = 0
        fig = figure(nrows * ncols, (20, 4 * nrows), layout="constrained")
        fig.suptitle(
            f"Reduction of CycleMask and IsNACColoring checks against the naive algorithm",
            fontsize=20,
        )
        figs.append(fig)

        for title, value_columns in zip(
            [
                "The number of checks",
                # "#is_NAC_coloring() calls/#NAC(G)",
                "The number of NAC-colorings / The number of checks",
                # "Count: metric / monochromatic classes number",
            ],
            ops_value_groups,
        ):
            axs = [
                fig.add_subplot(nrows, ncols, i + ncols * row + 1)
                for i in range(len(ops_aggregation))
            ]
            for ax, aggregation in zip(axs, ops_aggregation):
                _plot_is_NAC_coloring_calls_groups(
                    title,
                    df,
                    ax,
                    x_column,
                    value_columns,
                    aggregation,
                    legend_rename_dict=rename_dict,
                )
            row += 1

    return figs


###############################################################################
