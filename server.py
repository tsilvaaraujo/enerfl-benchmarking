import os
import time
from logging import INFO

from flwr.common import log
from flwr.common import ArrayRecord, ConfigRecord, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAdam, FedAdagrad, FedAvg, FedAvgM, FedProx

from llm_pipeline import (
    build_llm_config,
    get_llm_initial_arrays,
    get_llm_model,
    make_save_path,
    set_llm_state,
)
from enerfl import Profiler, profiled
from task import get_model, get_model_params

app = ServerApp()


def _is_llm_run() -> bool:
    model_name = os.environ.get("MODEL_NAME", "").strip()
    return model_name == "OpenLLaMA"


def _expected_clients() -> int:
    return int(os.environ.get("NUM_CLIENTS", "0") or 0)


def _wait_for_clients(grid: Grid, expected: int, timeout: float) -> None:
    """Block until every SuperNode is connected, before any round is timed."""
    if expected <= 0:
        return
    t0 = time.monotonic()
    last = -1
    while (connected := len(list(grid.get_node_ids()))) < expected:
        if connected != last:
            log(INFO, "Waiting for clients: %d/%d connected", connected, expected)
            last = connected
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"only {connected}/{expected} clients connected after {timeout:.0f}s")
        time.sleep(1)
    log(INFO, "All %d clients connected after %.1fs", expected, time.monotonic() - t0)


def _build_strategy(algo_name: str, fraction_train: float, fraction_eval: float):
    # Without these, Flower starts a round as soon as 2 nodes are connected.
    n = max(_expected_clients(), 2)
    common = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
        min_available_nodes=n,
        min_train_nodes=n if fraction_train > 0 else 0,
        min_evaluate_nodes=n if fraction_eval > 0 else 0,
    )
    if algo_name == "FedProx":
        return FedProx(**common, proximal_mu=0.1)
    if algo_name == "FedAvgM":
        return FedAvgM(**common, server_momentum=0.9)
    if algo_name == "FedAdagrad":
        return FedAdagrad(**common)
    if algo_name == "FedAdam":
        return FedAdam(**common)
    return FedAvg(**common)


def _get_llm_evaluate_fn(cfg, save_every_round: int, total_rounds: int, save_path: str):
    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round != 0 and (
            server_round == total_rounds
            or server_round % save_every_round == 0
        ):
            model = get_llm_model(cfg.model)
            set_llm_state(model, arrays.to_torch_state_dict())
            model.save_pretrained(f"{save_path}/peft_{server_round}")
        return MetricRecord()

    return evaluate


@app.main()
def main(grid: Grid, context) -> None:
    num_rounds = context.run_config.get("num-server-rounds", 5)
    round_timeout = float(context.run_config.get("round-timeout", 3600.0))
    fraction_train = context.run_config.get("fraction-train", 1.0)
    fraction_eval = context.run_config.get("fraction-evaluate", 0.0)
    profiler_enabled = bool(context.run_config.get("profiler-enabled", True))
    algo_name = os.environ.get("ALGO", "FedAvg")

    strategy = _build_strategy(algo_name, fraction_train, fraction_eval)

    profiler = Profiler(
        enabled=profiler_enabled,
        enable_timing=True,
        enable_update_count=True,
        enable_payload_size=True,
        enable_carbon=True,
        carbon_tracker_kwargs={"country_iso_code": "FRA"},
        stream_csv_path=os.environ.get("PROFILE_CSV"),
        network_interface=os.environ.get("NETWORK_INTERFACE"),
        algorithm_name=algo_name,
    )
    strategy = profiled(strategy, profiler)

    connect_timeout = float(context.run_config.get("connect-timeout", 1800.0))

    if _is_llm_run():
        cfg = build_llm_config(context.run_config)
        save_path = make_save_path()
        initial_arrays = ArrayRecord(get_llm_initial_arrays(cfg))
        evaluate_fn = _get_llm_evaluate_fn(
            cfg, cfg.train.save_every_round, num_rounds, save_path
        )
        _wait_for_clients(grid, _expected_clients(), connect_timeout)
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            timeout=round_timeout,
            train_config=ConfigRecord({"save_path": save_path}),
            evaluate_config=ConfigRecord(),
            evaluate_fn=evaluate_fn,
        )
        return

    model = get_model()
    initial_arrays = ArrayRecord(get_model_params(model))
    _wait_for_clients(grid, _expected_clients(), connect_timeout)
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=round_timeout,
        train_config=ConfigRecord(),
        evaluate_config=ConfigRecord(),
    )
