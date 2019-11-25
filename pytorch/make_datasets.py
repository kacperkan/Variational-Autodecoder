import typing as t
from pathlib import Path

import h5py
import tqdm
import numpy as np
import torchvision

np.random.seed(0)


def generate_mnist(train_ratio: float) -> t.Dict[str, np.ndarray]:
    train_mnist = torchvision.datasets.MNIST(
        "data/", download=True, train=True
    )
    test_mnist = torchvision.datasets.MNIST(
        "data/", download=True, train=False
    )

    x_train, y_train = train_mnist.data, train_mnist.targets
    x_test, y_test = train_mnist.data, train_mnist.targets

    indices = np.arange(0, len(x_train))
    np.random.shuffle(indices)

    split_point = int(len(indices) * train_ratio)
    train_indices = indices[:split_point]
    valid_indices = indices[split_point:]

    (x_train, y_train), (x_valid, y_valid) = (
        (x_train[train_indices], y_train[train_indices]),
        (x_train[valid_indices], y_train[valid_indices]),
    )

    output = {
        "train_2d": x_train,
        "valid_2d": x_valid,
        "test_2d": x_test,
        "train_y_true": y_train,
        "valid_y_true": y_valid,
        "test_y_true": y_test,
    }

    return output


def generate_mask(input_data: np.ndarray, probability: float) -> np.ndarray:
    return np.random.binomial(1, probability, size=input_data.shape)


def save_data(
    dataset: t.Dict[str, np.ndarray], noises: t.List[float], name: str
):
    output_path = Path("data/") / name
    output_path.mkdir(parents=True, exist_ok=True)
    for noise in tqdm.tqdm(noises):
        output = {
            "train_mask": generate_mask(dataset["train_2d"], noise),
            "valid_mask": generate_mask(dataset["valid_2d"], noise),
            "test_mask": generate_mask(dataset["test_2d"], noise),
            **dataset,
        }

        if noise <= 1e-5:
            noise_output_path = output_path / f"{name}.h5"
        else:
            noise_output_path = output_path / f"{name}_{noise:.1f}.h5"

        with h5py.File(noise_output_path.as_posix(), "w") as f:
            for key, data in output.items():
                f.create_dataset(key, dtype=np.float32, data=data)


def main():
    import argparse

    available_datasets = ["mnist"]

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=available_datasets)
    parser.add_argument(
        "--noises",
        nargs="*",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        type=list,
    )
    parser.add_argument("--train_ratio", default=0.8, type=float)

    args = parser.parse_args()

    if args.dataset == "mnist":
        dataset = generate_mnist(args.train_ratio)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    save_data(dataset, args.noises, args.dataset)


if __name__ == "__main__":
    main()
