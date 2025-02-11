from datasets import MetaModuloDataset, MetaBitConceptsDataset, Omniglot
from models import MLP, CNN, LSTM, Transformer
from constants import *

def init_misc(experiment, alphabet):
    if alphabet in ["ancient", "asian"] and experiment == "omniglot":
        alphabet = ALPHABETS[alphabet]
    else:
        alphabet = None
    channels = 3 if experiment == "concept" else 1
    bits = 4 if experiment == "concept" else 8
    n_output = 20 if experiment == "omniglot" else 1
    return alphabet, bits, channels, n_output


def init_dataset(experiment, model, data_type, skip, n_support=None, alphabet=None):
    if experiment == "mod":
        if n_support is None:
            train_dataset = MetaModuloDataset(
                n_tasks=10000,
                skip=skip,
                train=True,
                data=data_type,
                model=model,
            )
        test_dataset = MetaModuloDataset(
            n_tasks=20 if n_support is not None else 100,
            skip=skip,
            train=False,
            data=data_type,
            model=model,
            n_support=n_support,
        )
        val_dataset = MetaModuloDataset(
            n_tasks=20 if n_support is not None else 100,
            skip=skip,
            train=True,
            data=data_type,
            model=model,
            n_support=n_support,
        )
    elif experiment == "concept":
        if n_support is None:
            train_dataset = MetaBitConceptsDataset(
                n_tasks=10000,
                data=data_type,
                model=model,
            )
        test_dataset = MetaBitConceptsDataset(
            n_tasks=10 if n_support is not None else 100,
            data=data_type,
            model=model,
            n_support=n_support,
        )
        val_dataset = MetaBitConceptsDataset(
            n_tasks=10 if n_support is not None else 100,
            data=data_type,
            model=model,
            n_support=n_support,
        )
    elif experiment == "omniglot":
        if n_support is None:
            train_dataset = Omniglot(
                n_tasks=10000,
                model=model,
                train=True,
                alphabet=alphabet
            )
        test_dataset = Omniglot(
            n_tasks=10 if n_support is not None else 100,
            model=model,
            train=False,
        )
        val_dataset = Omniglot(
            n_tasks=10 if n_support is not None else 100,
            model=model,
            train=True,
            alphabet=alphabet
        )
    else:
        raise ValueError("Dataset Unrecognized.")

    if n_support is not None:
        return test_dataset, val_dataset
    else:
        return train_dataset, test_dataset, val_dataset


def init_model(
    m, data_type, index, verbose: bool = False, channels: int = 1, bits: int = 8, n_output: int = 1,
):
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
        model = MLP(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_input_channels=channels,
        )
    elif m == "cnn":
        n_hidden, n_layers = CNN_PARAMS[index]
        model = CNN(n_input_channels=channels, n_output=n_output, n_hiddens=n_hidden, n_layers=n_layers)
    elif m == "lstm":
        n_hidden, n_layers = LSTM_PARAMS[index]
        model = LSTM(n_input=n_input, n_output=n_output, n_hidden=n_hidden, n_layers=n_layers)
    elif m == "transformer":
        n_hidden, n_layers = TRANSFORMER_PARAMS[index]
        model = Transformer(
            n_input=n_input,
            n_output=n_output,
            d_model=n_hidden,
            dim_feedforward=2 * n_hidden,
            num_layers=n_layers,
        )
    else:
        raise ValueError("Model unrecognized.")

    if verbose:
        print(f"Initialized {m} with {n_hidden} hidden units and {n_layers} layers...")
    return model
