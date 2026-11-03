import math
import os
import tempfile
from datetime import datetime
from typing import Any

import torch
from flwr.common.config import unflatten_dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

FDS = None


def replace_keys(input_dict: dict[str, Any], match: str = "-", target: str = "_") -> dict[str, Any]:
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def build_llm_config(run_config: dict[str, Any]) -> DictConfig:
    cfg = DictConfig(replace_keys(unflatten_dict(run_config)))
    return cfg


def formatting_prompts_func(example: dict[str, Any]) -> str:
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    instruction = example["instruction"]
    inp = example.get("input", "")
    resp = example.get("response") or example.get("output") or ""
    if inp:
        return (
            f"{mssg}\n### Instruction:\n{instruction}\n"
            f"### Input:\n{inp}\n### Response: {resp}"
        )
    return (
        f"{mssg}\n### Instruction:\n{instruction}\n"
        f"### Response: {resp}"
    )


def get_tokenizer_and_data_collator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, formatting_prompts_func


def load_llm_data(partition_id: int, num_partitions: int, dataset_name: str):
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    client_data = FDS.load_partition(partition_id, "train")
    if "output" in client_data.column_names:
        client_data = client_data.rename_column("output", "response")
    split = client_data.train_test_split(test_size=0.2, seed=42)
    return split["train"], split["test"]


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_llm_model(model_cfg: DictConfig):
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            "Use 4-bit or 8-bit quantization. "
            f"You passed: {model_cfg.quantization}"
        )

    torch_dtype = getattr(model_cfg, "torch_dtype", None) or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)


def get_llm_initial_arrays(cfg: DictConfig):
    model = get_llm_model(cfg.model)
    return get_peft_model_state_dict(model)


def set_llm_state(model, arrays) -> None:
    set_peft_model_state_dict(model, arrays)


def build_training_arguments(cfg: DictConfig, save_path: str) -> SFTConfig:
    args_dict = dict(cfg.train.training_arguments)

    if not save_path:
        save_path = os.path.join(tempfile.gettempdir(), "enerfl-trainer")
    os.makedirs(save_path, exist_ok=True)
    args_dict["output_dir"] = save_path
    if not torch.cuda.is_available():
        args_dict.setdefault("bf16", False)
        args_dict.setdefault("fp16", False)
        args_dict.setdefault("use_cpu", True)
    return SFTConfig(**args_dict)


def make_save_path(base_dir: str = "results") -> str:
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"{base_dir}/{folder_name}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def train_llm(
    model,
    tokenizer,
    formatting_func,
    training_args: SFTConfig,
    trainset,
    seq_length: int,
):
    training_args.max_seq_length = seq_length
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=trainset,
        formatting_func=formatting_func,
    )
    return trainer.train()


def evaluate_llm(
    model,
    tokenizer,
    formatting_func,
    training_args: SFTConfig,
    evalset,
    seq_length: int,
):
    training_args.max_seq_length = seq_length
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=evalset,
        eval_dataset=evalset,
        formatting_func=formatting_func,
    )
    return trainer.evaluate()
