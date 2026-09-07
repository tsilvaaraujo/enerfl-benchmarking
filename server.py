import os

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


def _build_strategy(algo_name: str, fraction_train: float, fraction_eval: float):
    if algo_name == "FedProx":
        return FedProx(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
            proximal_mu=0.1,
        )
    if algo_name == "FedAvgM":
        return FedAvgM(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
            server_momentum=0.9,
        )
    if algo_name == "FedAdagrad":
        return FedAdagrad(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
        )
    if algo_name == "FedAdam":
        return FedAdam(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
        )
    return FedAvg(fraction_train=fraction_train, fraction_evaluate=fraction_eval)


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

    if _is_llm_run():
        cfg = build_llm_config(context.run_config)
        save_path = make_save_path()
        initial_arrays = ArrayRecord(get_llm_initial_arrays(cfg))
        evaluate_fn = _get_llm_evaluate_fn(
            cfg, cfg.train.save_every_round, num_rounds, save_path
        )
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
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        timeout=round_timeout,
        train_config=ConfigRecord(),
        evaluate_config=ConfigRecord(),
    )
