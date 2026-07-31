"""Training entrypoint for the Edge Impulse PyTorch keyword-spotting block.

Edge Impulse passes the following arguments automatically:

    --info-file <file>        Path to train_input.json (training configuration).
    --data-directory <dir>    Directory with the NumPy train/test splits.
    --out-directory <dir>     Directory to write model artifacts to.

Custom parameters defined in parameters.json are passed as extra arguments:

    --epochs <int>
    --learning-rate <float>
    --batch-size <int>
    --columns <int>           Coefficients per frame (MFE filters / MFCC coeffs).
    --export-pte <flag>

Input data (already processed by an Edge Impulse MFE or MFCC block):

    X_split_train.npy / X_split_test.npy   float32, shape (batch, features)
    Y_split_train.npy / Y_split_test.npy   one-hot label matrix, shape (batch, num_classes)

MFE/MFCC blocks flatten each window row-major as
[frame0_coeff0, frame0_coeff1, ..., frame1_coeff0, ...]. Reshaping to
(frames, columns) and adding a channel dimension recovers the spectrogram the
CNN expects. Because we export ONNX, Edge Impulse handles the equivalent
reshape on-device automatically.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import ClassifierExport, build_model

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch keyword-spotting learning block")
    parser.add_argument("--info-file", type=str, required=False)
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--out-directory", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--columns", type=int, default=40)
    parser.add_argument("--export-pte", action="store_true", default=False)
    # Ignore any unknown args Edge Impulse may add in the future.
    args, _ = parser.parse_known_args()
    return args


def load_split(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(os.path.join(data_dir, f"X_split_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"Y_split_{split}.npy"))
    return x, y


def to_labels(y: np.ndarray) -> np.ndarray:
    """Convert Edge Impulse's one-hot Y split to integer class indices.

    Edge Impulse encodes classification labels as a one-hot matrix of shape
    (n_samples, n_classes). PyTorch's CrossEntropyLoss expects integer targets,
    so we take the argmax across the class dimension.
    """
    y = np.asarray(y)
    if not y.size:
        return np.array([], dtype=np.int64)
    return y.argmax(axis=1).astype(np.int64)


def to_spectrogram_nchw(x: np.ndarray, columns: int) -> np.ndarray:
    """Reshape flat audio features to a single-channel NCHW spectrogram.

    Handles the shapes Edge Impulse might deliver:

    - 2D ``(batch, features)`` -> reshape to ``(batch, 1, frames, columns)``.
    - 3D ``(batch, frames, columns)`` -> add the channel dimension.
    - 4D ``(batch, height, width, channels)`` (NHWC) -> transpose to NCHW.

    We force a C-contiguous copy so torch.export captures dim_order
    (0, 1, 2, 3); the XNNPACK backend rejects channels-last inputs during
    .pte lowering.
    """
    if x.ndim == 4:
        return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    if x.ndim == 3:
        return np.ascontiguousarray(x[:, None, :, :])
    if x.ndim == 2:
        n_features = x.shape[1]
        if columns <= 0 or n_features % columns != 0:
            raise ValueError(
                f"Feature count {n_features} is not divisible by columns {columns}. "
                "Set --columns to your MFE filter number (40) or MFCC coefficient "
                "count (13)."
            )
        frames = n_features // columns
        return np.ascontiguousarray(x.reshape(x.shape[0], 1, frames, columns))
    raise ValueError(
        f"Expected 2D/3D/4D feature data, got shape {x.shape}. "
        "This block operates on MFE/MFCC audio features."
    )


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    tensor_x = torch.from_numpy(x.astype(np.float32))
    tensor_y = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == targets).sum().item()
    return correct / len(loader.dataset)


def export_onnx(model: nn.Module, sample_input: torch.Tensor, out_path: str) -> None:
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Saved ONNX model to {out_path}")


def export_pte(model: nn.Module, sample_input: torch.Tensor, out_path: str) -> None:
    """Export an ExecuTorch program (.pte) lowered to the XNNPACK backend.

    ExecuTorch is optional at runtime; if it isn't installed we warn and skip
    rather than failing the whole training job.
    """
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower
    except ImportError:
        print(
            "WARNING: executorch is not installed; skipping .pte export. "
            "Add 'executorch' to requirements.txt to enable it."
        )
        return

    model.eval()
    # XNNPACK only accepts contiguous (channels-first) inputs during lowering.
    sample_input = sample_input.contiguous(memory_format=torch.contiguous_format)
    try:
        exported = torch.export.export(model, (sample_input,))
        program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
    except Exception as exc:  # noqa: BLE001 - .pte is an optional artifact
        print(
            "WARNING: ExecuTorch lowering failed; skipping .pte export "
            f"(the ONNX model is unaffected). Reason: {exc}"
        )
        return
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    print(f"Saved ExecuTorch program to {out_path}")


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, y_train = load_split(args.data_directory, "train")
    x_test, y_test = load_split(args.data_directory, "test")

    x_train = to_spectrogram_nchw(x_train, args.columns)
    x_test = (
        to_spectrogram_nchw(x_test, args.columns)
        if x_test.size
        else x_test.reshape(0, *x_train.shape[1:])
    )

    in_channels = x_train.shape[1]
    num_classes = int(y_train.shape[1])
    label_train = to_labels(y_train)
    label_test = to_labels(y_test)
    print(
        f"Spectrogram: {x_train.shape[1:]} (C,H,W), classes: {num_classes}, "
        f"train: {len(x_train)}, test: {len(x_test)}"
    )

    train_loader = make_loader(x_train, label_train, args.batch_size, shuffle=True)
    test_loader = make_loader(x_test, label_test, args.batch_size, shuffle=False)

    model = build_model(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.4f} - val_acc: {acc:.4f}")

    os.makedirs(args.out_directory, exist_ok=True)
    model = model.to("cpu")
    export_model = ClassifierExport(model)
    # ascontiguousarray guarantees dim_order (0, 1, 2, 3) for the XNNPACK export.
    sample_input = torch.from_numpy(
        np.ascontiguousarray(x_train[:1], dtype=np.float32)
    )

    export_onnx(export_model, sample_input, os.path.join(args.out_directory, "model.onnx"))
    if args.export_pte:
        export_pte(export_model, sample_input, os.path.join(args.out_directory, "model.pte"))


if __name__ == "__main__":
    main()
