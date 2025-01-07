from datasets import MetaModuloDataset, MetaBitConceptsDataset
from models import MLP, CNN, LSTM, Transformer
from constants import *

def init_dataset(experiment, model, data_type, n_samples_per_task, skip):
    if experiment == "mod":
        train_dataset = MetaModuloDataset(
            n_samples_per_task=n_samples_per_task,
            skip=skip,
            train=True,
            data=data_type,
            model=model,
        )
        test_dataset = MetaModuloDataset(
            n_samples_per_task=n_samples_per_task,
            skip=skip,
            train=False,
            data=data_type,
            model=model,
        )
        val_dataset = MetaModuloDataset(
            n_tasks=20,
            n_samples_per_task=n_samples_per_task,
            skip=skip,
            train=True,
            data=data_type,
            model=model,
        )
    elif experiment == "concept":
        train_dataset = MetaBitConceptsDataset(
            n_tasks=1000,
            data=data_type,
            model=model,
        )
        test_dataset = MetaBitConceptsDataset(
            n_tasks=100,
            data=data_type,
            model=model,
        )
        val_dataset = MetaBitConceptsDataset(
            n_tasks=100,
            data=data_type,
            model=model,
        )
    else:
        raise ValueError

    return train_dataset, test_dataset, val_dataset

def init_model(m, data_type, index, verbose: bool = False, channels: int = 1, bits: int = 8):
    if data_type == "image":
        n_input = 32 * 32 * channels if m == "mlp" else 16 * channels
    elif data_type == "bits":
        n_input = bits if m == "mlp" else 1
    elif data_type == "number":
        n_input = 1
    else:
        raise ValueError("Data Type unrecognized.")

    if m == "mlp":
        n_hidden, n_layers = MLP_PARAMS[index]
        model = MLP(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
    elif m == "cnn":
        n_hidden, n_layers = CNN_PARAMS[index]
        model = CNN(n_input_channels=channels, n_hiddens=n_hidden, n_layers=n_layers)
    elif m == "lstm":
        n_hidden, n_layers = LSTM_PARAMS[index]
        model = LSTM(n_input=n_input, n_hidden=n_hidden, n_layers=n_layers)
    elif m == "transformer":
        n_hidden, n_layers = TRANSFORMER_PARAMS[index]
        model = Transformer(
            n_input=n_input,
            d_model=n_hidden,
            dim_feedforward=2 * n_hidden,
            num_layers=n_layers,
        )
    else:
        raise ValueError("Model unrecognized.")

    if verbose:
        print(f"Initialized {m} with {n_hidden} hidden units and {n_layers} layers...")
    return model