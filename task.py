"""EnerFL data and model factory based on environment variables."""
import os

import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer

VISION_DATASETS = {"MNIST", "FEMNIST", "CIFAR-10"}
TEXT_DATASETS = {"Sentiment140"}
AUDIO_DATASETS = {"Ambient Acoustic Context"}
LLM_DATASETS = {"Alpaca-GPT4"}

VISION_MODELS = {"EfficientNet-B0", "ResNet-50", "VGG-16", "ViT-B/16"}
SEQUENCE_MODELS = {"LSTM"}
TOKEN_MODEL = "OpenLLaMA"

DATASET_REGISTRY = {
    "CIFAR-10": {"hf_id": "uoft-cs/cifar10", "image_key": "img", "label_key": "label"},
    "MNIST": {"hf_id": "ylecun/mnist", "image_key": "image", "label_key": "label"},
    "FEMNIST": {"hf_id": "flwrlabs/femnist", "image_key": "image", "label_key": "character"},
    "Sentiment140": {"hf_id": "stanfordnlp/sentiment140", "label_key": "sentiment"},
    "Ambient Acoustic Context": {
        "hf_id": "speech_commands",
        "subset": "v0.02",
        "audio_key": "audio",
        "label_key": "label",
    },
    "Alpaca-GPT4": {"hf_id": "flwrlabs/alpaca-gpt4"},
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _infer_num_classes(dataset_name: str) -> int:
    override = os.environ.get("NUM_CLASSES")
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass

    mapping = {
        "MNIST": 10,
        "FEMNIST": 62,
        "CIFAR-10": 10,
        "Sentiment140": 2,
        "Ambient Acoustic Context": 35,
        "Alpaca-GPT4": 2,
    }
    return mapping.get(dataset_name, 10)


def _vision_image_size(model_name: str, dataset_name: str) -> int:
    if model_name in VISION_MODELS:
        return 224
    if dataset_name == "CIFAR-10":
        return 32
    return 28


def _make_vision_transform(dataset_name: str, model_name: str) -> transforms.Compose:
    image_size = _vision_image_size(model_name, dataset_name)
    to_three = dataset_name in {"MNIST", "FEMNIST"} and model_name in VISION_MODELS

    ops = [transforms.Resize((image_size, image_size))]
    if to_three:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())

    if dataset_name == "CIFAR-10" or to_three:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        mean = (0.1307,)
        std = (0.3081,)
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def _pad_or_trim_1d(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    if audio.numel() >= target_len:
        return audio[:target_len]
    pad = torch.zeros(target_len - audio.numel(), dtype=audio.dtype)
    return torch.cat([audio, pad])


def _audio_to_sequence(audio: torch.Tensor, seq_len: int, feature_dim: int) -> torch.Tensor:
    total = seq_len * feature_dim
    audio = _pad_or_trim_1d(audio, total)
    return audio.view(seq_len, feature_dim)


def _vision_to_sequence(img: torch.Tensor, seq_len: int, feature_dim: int) -> torch.Tensor:
    flat = img.view(-1)
    total = seq_len * feature_dim
    if flat.numel() < total:
        pad = torch.zeros(total - flat.numel(), dtype=flat.dtype)
        flat = torch.cat([flat, pad])
    flat = flat[:total]
    return flat.view(seq_len, feature_dim)


def get_model(
    model_name: str | None = None,
    dataset_name: str | None = None,
    num_classes: int | None = None,
):
    dataset_name = dataset_name or os.environ.get("DATASET", "CIFAR-10")
    model_name = model_name or os.environ.get("MODEL_NAME", "EfficientNet-B0")
    num_classes = num_classes or _infer_num_classes(dataset_name)

    use_pretrained = _env_bool("USE_PRETRAINED", False)

    if model_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet-50":
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "VGG-16":
        weights = models.VGG16_Weights.DEFAULT if use_pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "ViT-B/16":
        weights = models.ViT_B_16_Weights.DEFAULT if use_pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "LSTM":
        hidden_dim = _env_int("LSTM_HIDDEN_DIM", 256)
        num_layers = _env_int("LSTM_NUM_LAYERS", 2)

        if dataset_name in TEXT_DATASETS:
            vocab_size = _env_int("TEXT_VOCAB_SIZE", 30522)
            embed_dim = _env_int("LSTM_EMBED_DIM", 128)

            class TextLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(
                        embed_dim, hidden_dim, num_layers, batch_first=True
                    )
                    self.fc = nn.Linear(hidden_dim, num_classes)

                def forward(self, x):
                    x = self.embedding(x)
                    _, (hn, _) = self.lstm(x)
                    return self.fc(hn[-1])

            model = TextLSTM()
        else:
            input_dim = _env_int("LSTM_INPUT_DIM", 100)

            class SequenceLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_dim, hidden_dim, num_layers, batch_first=True
                    )
                    self.fc = nn.Linear(hidden_dim, num_classes)

                def forward(self, x):
                    _, (hn, _) = self.lstm(x)
                    return self.fc(hn[-1])

            model = SequenceLSTM()
    elif model_name == TOKEN_MODEL:
        raise ValueError("OpenLLaMA runs via the LLM pipeline, not task.get_model().")
    else:
        use_grayscale = dataset_name in {"MNIST", "FEMNIST"}
        in_channels = 1 if use_grayscale else 3
        image_size = _vision_image_size(model_name, dataset_name)
        flat_dim = 64 * (image_size // 2 - 2) * (image_size // 2 - 2)

        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.fc1 = nn.Linear(flat_dim, 128)
                self.fc2 = nn.Linear(128, num_classes)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(self.conv2(x), 2)
                x = torch.flatten(x, 1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleCNN()

    return model


def _get_federated_dataset(dataset_name: str, num_partitions: int) -> FederatedDataset:
    spec = DATASET_REGISTRY[dataset_name]
    if dataset_name in AUDIO_DATASETS:
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="speaker_id", group_size=5
        )
    else:
        partitioner = IidPartitioner(num_partitions=num_partitions)

    return FederatedDataset(
        dataset=spec["hf_id"],
        subset=spec.get("subset"),
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def load_data(partition_id, num_partitions, batch_size=32):
    dataset_name = os.environ.get("DATASET", "CIFAR-10")
    model_name = os.environ.get("MODEL_NAME", "EfficientNet-B0")
    num_classes = _infer_num_classes(dataset_name)

    if dataset_name in LLM_DATASETS or model_name == TOKEN_MODEL:
        raise ValueError("LLM datasets must be loaded via llm_pipeline.")

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    fds = _get_federated_dataset(dataset_name, num_partitions)
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if dataset_name in VISION_DATASETS:
        spec = DATASET_REGISTRY[dataset_name]
        image_key = spec.get("image_key", "image")
        label_key = spec.get("label_key", "label")
        transform = _make_vision_transform(dataset_name, model_name)
        seq_len = _env_int("SEQ_LEN", 32)
        feature_dim = _env_int("LSTM_INPUT_DIM", 100)

        def apply_vision_transforms(batch):
            images = batch[image_key]
            inputs = [transform(img) for img in images]
            if model_name in SEQUENCE_MODELS:
                inputs = [
                    _vision_to_sequence(img, seq_len, feature_dim) for img in inputs
                ]
            labels = batch.get(label_key)
            if labels is None:
                labels = batch.get("label") or batch.get("labels")
            if labels is None:
                raise ValueError(
                    f"Vision dataset {dataset_name} is missing label column '{label_key}'."
                )
            return {"inputs": inputs, "labels": labels}

        partition_train_test = partition_train_test.with_transform(
            apply_vision_transforms
        )
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

    if dataset_name in TEXT_DATASETS:
        spec = DATASET_REGISTRY[dataset_name]
        label_key = spec.get("label_key", "label")
        tokenizer_name = os.environ.get("TEXT_TOKENIZER", "bert-base-uncased")
        max_len = _env_int("TEXT_MAX_LEN", 128)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, model_max_length=max_len
        )

        def tokenize_function(batch):
            texts = batch.get("text") or batch.get("sentence")
            tokens = tokenizer(
                texts, truncation=True, padding="max_length", max_length=max_len
            )
            labels = batch.get(label_key)
            if labels is None:
                labels = batch.get("label") or batch.get("labels")
            if labels is None:
                raise ValueError(
                    f"Text dataset {dataset_name} is missing label column '{label_key}'."
                )
            labels = [int(val) for val in labels]
            labels = [1 if val > 0 else 0 for val in labels]
            tokens["labels"] = labels
            tokens["inputs"] = tokens["input_ids"]
            return tokens

        partition_train_test = partition_train_test.map(
            tokenize_function, batched=True
        )
        partition_train_test = partition_train_test.remove_columns(
            [
                col
                for col in partition_train_test["train"].column_names
                if col not in {"inputs", "labels"}
            ]
        )
        partition_train_test.set_format(type="torch")
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

    if dataset_name in AUDIO_DATASETS:
        spec = DATASET_REGISTRY[dataset_name]
        audio_key = spec.get("audio_key", "audio")
        label_key = spec.get("label_key", "label")
        audio_samples = _env_int("AUDIO_SAMPLES", 16000)
        seq_len = _env_int("AUDIO_SEQ_LEN", 64)
        feature_dim = _env_int("AUDIO_FEATURE_DIM", 64)
        freq_bins = _env_int("AUDIO_FREQ_BINS", 64)

        def apply_audio_transforms(batch):
            audio_batch = batch[audio_key]
            inputs = []
            for item in audio_batch:
                array = torch.tensor(item["array"], dtype=torch.float32)
                array = _pad_or_trim_1d(array, audio_samples)
                if model_name in SEQUENCE_MODELS:
                    array = _audio_to_sequence(array, seq_len, feature_dim)
                elif model_name in VISION_MODELS:
                    total = (audio_samples // freq_bins) * freq_bins
                    array = array[:total].view(1, freq_bins, total // freq_bins)
                else:
                    array = array.unsqueeze(0)
                inputs.append(array)
            labels = batch.get(label_key)
            if labels is None:
                labels = batch.get("label") or batch.get("labels")
            if labels is None:
                raise ValueError(
                    f"Audio dataset {dataset_name} is missing label column '{label_key}'."
                )
            return {"inputs": inputs, "labels": labels}

        partition_train_test = partition_train_test.with_transform(
            apply_audio_transforms
        )
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

    raise ValueError(f"Dataset {dataset_name} is not supported.")

def get_model_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_params(model, params):
    state_dict = model.state_dict()
    for (name, tensor), param in zip(state_dict.items(), params):
        state_dict[name] = torch.tensor(param)
    model.load_state_dict(state_dict)
