# PyTorch keyword spotting learning block (ExecuTorch)

An [Edge Impulse custom learning block](https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks) that trains a PyTorch keyword-spotting classifier on **audio** features and exports:

- `model.onnx` — consumed by Edge Impulse (auto-converted to TFLite for deployment).
- `model.pte` — an optional [ExecuTorch](https://pytorch.org/executorch/) program for native PyTorch on-device inference.

This is the audio / keyword-spotting variant of the ExecuTorch end-to-end series (see also the image, time-series and object-detection blocks).

Compatible with the [Executorch Deploy Block](https://github.com/edgeimpulse/executorch-deploy)

## Files

| File | Purpose |
|------|---------|
| `parameters.json` | Block metadata and the parameters shown in Studio (epochs, learning rate, batch size, feature columns, `.pte` export flag). |
| `model.py` | `KwsCNN` — a compact 2D CNN over the MFE/MFCC spectrogram. |
| `train.py` | Training entrypoint. Loads the NumPy splits, reshapes to a spectrogram, trains, and exports ONNX (+ `.pte`). |
| `Dockerfile` | Builds the training container. |
| `requirements.txt` | Pinned Python dependencies. |
| `tests/test_model.py` | Smoke tests for model shapes and data handling. |

## How it works

This block operates on `audio` data, so it pairs with an **MFE** (Mel-filterbank energy) or **MFCC** processing block. Edge Impulse provides each one-second window as a flat float32 feature vector. The block:

1. Loads `X_split_{train,test}.npy` (shape `(batch, features)`) and `Y_split_{train,test}.npy` (a one-hot matrix of shape `(batch, num_classes)`; the class count comes from its width and integer labels from `argmax`) from `--data-directory`.
2. Reshapes each flat window into a single-channel spectrogram `(1, frames, columns)`, where **columns** is the coefficients-per-frame parameter (40 for MFE, 13 for MFCC). MFE/MFCC blocks flatten row-major as `[frame0_coeff0, frame0_coeff1, ..., frame1_coeff0, ...]`, so `features / columns` recovers the frame count.
3. Trains a compact 2D CNN with Adam + cross-entropy. Adaptive average pooling keeps the classifier independent of the exact spectrogram size.
4. Writes `model.onnx` (and `model.pte` when `--export-pte` is set) to `--out-directory`.

Because we export ONNX, Edge Impulse handles the equivalent reshape/transpose on-device automatically.

## Test locally

Download data from a project with an audio (MFE/MFCC) impulse, then run the block:

```bash
# Pull the latest processed data into ./input
edge-impulse-blocks runner --download-data input/

# Build and run the training container
docker build -t executorch-pytorch-kws-block .
docker run --rm -v "$PWD":/app executorch-pytorch-kws-block \
    --data-directory /app/input \
    --out-directory /app/out \
    --epochs 30 --learning-rate 0.001 --batch-size 32 \
    --columns 40 --export-pte
```

Set `--columns 13` if your impulse uses an MFCC block instead of MFE.

After training you'll find `out/model.onnx` (and `out/model.pte`).

Run the unit tests without Docker:

```bash
pip install -r requirements.txt pytest
python -m pytest
```

## Push to Edge Impulse

```bash
edge-impulse-blocks init   # first time only
edge-impulse-blocks push
```

The block then appears under **Create impulse → Add learning block** in Studio.

## Walkthrough in Studio

Built and validated against a 3-class keyword-spotting project (`background_noise` / `hey_edge` / `unknown`) using an MFE block (40 filters, 0.02 s frames, 0.01 s stride → 99 frames × 40 = 3960 features per window) and this learning block.

## Notes

- **columns** must divide the feature count. For an MFE block it equals the filter number (40 by default); for MFCC it equals the coefficient count (13 by default). If it doesn't divide evenly, training stops with a clear error.
- The `.pte` file is **not** used by the Edge Impulse SDK. It's included so you can carry the same trained weights into a native ExecuTorch runtime (see the `executorch-deploy` block and the Android demo).
- Pin `torch` and `executorch` to compatible versions. If ExecuTorch isn't installed, `.pte` export is skipped with a warning instead of failing the job.
