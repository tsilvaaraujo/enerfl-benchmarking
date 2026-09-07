import os
import torch
import torch.nn as nn
from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, Context, Message, MetricRecord, RecordDict
from peft import get_peft_model_state_dict

from llm_pipeline import (
    build_training_arguments,
    build_llm_config,
    cosine_annealing,
    evaluate_llm,
    get_llm_model,
    get_tokenizer_and_data_collator,
    load_llm_data,
    set_llm_state,
    train_llm,
)
from enerfl import ClientProfiler
from task import get_model, load_data, get_model_params, set_model_params

app = ClientApp()


def _eager_carbon_init() -> None:
    if os.environ.get("PROFILER_MODE", "on").lower() in ("off", "false", "0"):
        return
    from enerfl.carbon import get_global_tracker

    get_global_tracker(country_iso_code="FRA")


_eager_carbon_init()


def _is_llm_run() -> bool:
    model_name = os.environ.get("MODEL_NAME", "").strip()
    return model_name == "OpenLLaMA"


def _extract_batch(batch, device):
    if isinstance(batch, (list, tuple)):
        inputs, labels = batch
    elif isinstance(batch, dict):
        inputs = batch.get("inputs")
        if inputs is None:
            inputs = batch.get("input_ids")
        labels = batch.get("labels")
        if labels is None:
            labels = batch.get("label")
    else:
        raise ValueError("Unsupported batch type for training.")
    return inputs.to(device), labels.to(device)

@app.train()
def train(msg: Message, context: Context):
    show_profiler_summary = os.environ.get("PROFILER_SUMMARY", "").lower() in (
        "1",
        "true",
        "on",
    )
    algo_name = os.environ.get("ALGO", "FedAvg")
    profiler_enabled = bool(context.run_config.get("profiler-enabled", True))

    if _is_llm_run():
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        cp = ClientProfiler(
            enable_carbon=True,
            algorithm_name=algo_name,
            enabled=profiler_enabled,
            country_iso_code="FRA",
        )

        cfg = build_llm_config(context.run_config)
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 1)
        num_rounds = context.run_config.get("num-server-rounds", 1)

        with cp.stage("data_loading"):
            trainset, _ = load_llm_data(partition_id, num_partitions, cfg.dataset.name)
            tokenizer, formatting_func = get_tokenizer_and_data_collator(cfg.model.name)

        with cp.stage("model_creation"):
            model = get_llm_model(cfg.model)

        with cp.stage("model_deserialization"):
            set_llm_state(model, msg.content["arrays"].to_torch_state_dict())

        with cp.stage("local_training"):
            new_lr = cosine_annealing(
                msg.content["config"]["server-round"],
                num_rounds,
                cfg.train.learning_rate_max,
                cfg.train.learning_rate_min,
            )
            training_args = build_training_arguments(
                cfg, msg.content["config"].get("save_path", "")
            )
            training_args.learning_rate = new_lr
            results = train_llm(
                model=model,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
                training_args=training_args,
                trainset=trainset,
                seq_length=cfg.train.seq_length,
            )

        with cp.stage("model_serialization"):
            peft_state = {
                k: v.detach().cpu().to(torch.float32)
                for k, v in get_peft_model_state_dict(model).items()
            }
            model_record = ArrayRecord(peft_state)

        metrics = {
            "num-examples": len(trainset),
            "train_loss": getattr(results, "training_loss", 0.0),
        }
        metrics.update(cp.to_metric_record())
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})

        if show_profiler_summary:
            cp.summary()

        return Message(content=content, reply_to=msg)

    cp = ClientProfiler(
        enable_carbon=True,
        algorithm_name=algo_name,
        enabled=profiler_enabled,
        country_iso_code="FRA",
    )

    with cp.stage("model_creation"):
        model = get_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    with cp.stage("model_deserialization"):
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        set_model_params(model, ndarrays)

    global_params = [p.detach().clone() for p in model.parameters()]

    with cp.stage("data_loading"):
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 1)
        trainloader, _ = load_data(partition_id, num_partitions)

    with cp.stage("local_training"):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        proximal_mu = msg.content["config"].get("proximal-mu", 0.0)
        for batch in trainloader:
            inputs, labels = _extract_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if proximal_mu > 0:
                proximal_term = 0.0
                for local_param, global_param in zip(
                    model.parameters(), global_params
                ):
                    proximal_term += (local_param - global_param).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

    with cp.stage("model_serialization"):
        ndarrays = get_model_params(model)
        model_record = ArrayRecord(ndarrays)

    metrics = {
        "num-examples": len(trainloader.dataset),
    }
    metrics.update(cp.to_metric_record())
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    if show_profiler_summary:
        cp.summary()

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    show_profiler_summary = False
    algo_name = os.environ.get("ALGO", "FedAvg")
    profiler_enabled = bool(context.run_config.get("profiler-enabled", True))

    if _is_llm_run():
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        cp = ClientProfiler(
            enable_carbon=True,
            algorithm_name=algo_name,
            enabled=profiler_enabled,
            country_iso_code="FRA",
        )

        cfg = build_llm_config(context.run_config)
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 1)

        with cp.stage("data_loading"):
            _, evalset = load_llm_data(partition_id, num_partitions, cfg.dataset.name)
            tokenizer, formatting_func = get_tokenizer_and_data_collator(cfg.model.name)

        with cp.stage("model_creation"):
            model = get_llm_model(cfg.model)

        with cp.stage("model_deserialization"):
            set_llm_state(model, msg.content["arrays"].to_torch_state_dict())

        with cp.stage("local_evaluation"):
            training_args = build_training_arguments(cfg, "")
            eval_metrics = evaluate_llm(
                model=model,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
                training_args=training_args,
                evalset=evalset,
                seq_length=cfg.train.seq_length,
            )
            eval_loss = float(eval_metrics.get("eval_loss", 0.0))

        metrics = {
            "num-examples": len(evalset),
            "test_logloss": eval_loss,
        }
        metrics.update(cp.to_metric_record())
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})

        if show_profiler_summary:
            cp.summary()

        return Message(content=content, reply_to=msg)

    cp = ClientProfiler(
        enable_carbon=True,
        algorithm_name=algo_name,
        enabled=profiler_enabled,
        country_iso_code="FRA",
    )

    with cp.stage("model_creation"):
        model = get_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    with cp.stage("model_deserialization"):
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        set_model_params(model, ndarrays)

    with cp.stage("data_loading"):
        partition_id = context.node_config.get("partition-id", 0)
        num_partitions = context.node_config.get("num-partitions", 1)
        _, testloader = load_data(partition_id, num_partitions)

    with cp.stage("local_evaluation"):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in testloader:
                inputs, labels = _extract_batch(batch, device)
                outputs = model(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss = loss / len(testloader)
        accuracy = correct / total

    metrics = {
        "num-examples": total,
        "test_logloss": loss,
        "accuracy": accuracy,
    }
    metrics.update(cp.to_metric_record())
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    if show_profiler_summary:
        cp.summary()

    return Message(content=content, reply_to=msg)
