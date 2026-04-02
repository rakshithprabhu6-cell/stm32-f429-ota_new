import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Paths (relative — runs inside GitHub Actions) ────────
CORRECTIONS = Path("corrections")
MODEL_PATH  = Path("model/mnist.keras")
BASE_PATH   = Path("model/mnist_base.keras")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUM_CLASSES       = 11
MNIST_PER_DIGIT   = 1000   # digits per class  → 10,000 total
LETTERS_PER_CLASS = 200    # per A-Z letter    →  5,200 total invalid


def make_class_weights(y_train):
    """
    Equal-total-loss weighting.

    Goal: total loss contribution of invalid class == total loss of ONE digit class.
    Formula: w_invalid = n_per_digit / n_invalid

    WHY NOT compute_class_weight('balanced'):
      With 1000 samples per digit and 4700 invalid, 'balanced' gives
      invalid a weight of 0.28 — DOWNWEIGHTING it. The model barely
      penalises invalid mistakes and collapses to predicting everything
      as invalid (or ignores the class entirely).

    WHY NOT hardcoded cw[10] = 15.0:
      Massive over-penalty → model predicts everything as invalid
      to minimise the enormous loss on those samples.

    THIS formula: scales automatically as dataset grows with corrections.
    """
    n_per_digit = int(np.sum(y_train == 0))   # one digit class count
    n_invalid   = int(np.sum(y_train == 10))

    if n_invalid == 0 or n_per_digit == 0:
        return {i: 1.0 for i in range(NUM_CLASSES)}

    w_invalid = n_per_digit / n_invalid
    cw = {i: 1.0 for i in range(10)}
    cw[10] = w_invalid

    print("\n      Class weights (equal-total-loss formula):")
    print(f"        Digits 0-9  : 1.0000  ({n_per_digit} samples each)")
    print(f"        Invalid(10) : {w_invalid:.4f}  ({n_invalid} samples)")
    print(f"        Loss check  -> digit: {n_per_digit * 1.0:.0f}  "
          f"invalid: {n_invalid * w_invalid:.0f}  (should match)")
    return cw


def load_corrections():
    """Load all .npy correction files from corrections/ folder."""
    cx, cy = [], []
    for f in sorted(CORRECTIONS.glob("*.npy")):
        try:
            pixels = np.load(str(f), allow_pickle=True).astype("float32")
            if pixels.ndim == 2:
                pixels = pixels[..., np.newaxis]
            elif pixels.shape == (784,):
                pixels = pixels.reshape(28, 28, 1)
            pixels = pixels / 255.0 if pixels.max() > 1.0 else pixels

            stem = f.stem
            if stem.startswith("invalid") or stem.startswith("label10"):
                label = 10
            elif stem.startswith("label"):
                label = int(stem.split("_")[0].replace("label", ""))
            else:
                print(f"  Skipping unknown file: {f.name}")
                continue

            if 0 <= label <= 10:
                cx.append(pixels)
                cy.append(label)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    return (np.array(cx), np.array(cy, dtype=np.int32)) if cx \
           else (np.array([]).reshape(0, 28, 28, 1), np.array([], dtype=np.int32))


def load_invalid_samples():
    """Load A-Z letters as class 10. Falls back to EMNIST then noise."""
    CSV_LOCAL = r"C:\Users\HP\Downloads\archive (8)\A_Z Handwritten Data\A_Z Handwritten Data.csv"
    CSV_CLOUD = "A_Z Handwritten Data.csv"
    CSV = CSV_LOCAL if os.path.exists(CSV_LOCAL) else \
          CSV_CLOUD  if os.path.exists(CSV_CLOUD) else None

    if CSV:
        print("      Loading A-Z CSV...")
        import pandas as pd
        df      = pd.read_csv(CSV, header=None)
        lbl_col = df.iloc[:, 0].values.astype(np.int32)
        pixels  = df.iloc[:, 1:].values.astype("float32") / 255.0
        x_az    = pixels.reshape(-1, 28, 28, 1)
        x_inv, y_inv = [], []
        for cls in range(26):
            idx = np.where(lbl_col == cls)[0][:LETTERS_PER_CLASS]
            x_inv.append(x_az[idx])
            y_inv.append(np.full(len(idx), 10, dtype=np.int32))
        x_inv = np.concatenate(x_inv)
        y_inv = np.concatenate(y_inv)
        print(f"      A-Z letters: {len(x_inv)}  ({LETTERS_PER_CLASS}x26)")
        return x_inv, y_inv

    print("      CSV not found — trying EMNIST...")
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/letters", split="train", as_supervised=True)
        x_inv, y_inv = [], []
        for img, _ in ds.take(MNIST_PER_DIGIT * 10):
            x_inv.append(img.numpy().astype("float32") / 255.0)
            y_inv.append(10)
        x_inv = np.array(x_inv)
        y_inv = np.array(y_inv, dtype=np.int32)
        print(f"      EMNIST: {len(x_inv)}")
        return x_inv, y_inv
    except Exception as e:
        print(f"      EMNIST failed ({e}) — using noise")
        n     = MNIST_PER_DIGIT * 10
        x_inv = np.random.rand(n, 28, 28, 1).astype("float32")
        y_inv = np.full(n, 10, dtype=np.int32)
        print(f"      Noise fallback: {n}")
        return x_inv, y_inv


def retrain_model():
    print("=" * 50)
    print("  Cloud Retrain — GitHub Actions")
    print("=" * 50)

    # ── [1] Load MNIST ───────────────────────────────────
    print(f"\n[1/4] Loading MNIST ({MNIST_PER_DIGIT} per digit)...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = x_tr.astype("float32")[..., None] / 255.0
    x_te = x_te.astype("float32")[..., None] / 255.0

    sx, sy = [], []
    for d in range(10):
        idx = np.where(y_tr == d)[0][:MNIST_PER_DIGIT]
        sx.append(x_tr[idx])
        sy.append(y_tr[idx])
    sx = np.concatenate(sx)
    sy = np.concatenate(sy).astype(np.int32)
    print(f"      MNIST: {len(sx)} samples")

    # ── [2] Load corrections ─────────────────────────────
    print("\n[2/4] Loading corrections...")
    cx, cy = load_corrections()
    digit_count   = int(np.sum(cy < 10))  if len(cy) else 0
    invalid_count = int(np.sum(cy == 10)) if len(cy) else 0
    print(f"      Digit corrections  : {digit_count}")
    print(f"      Invalid corrections: {invalid_count}")

    if len(cx) > 0:
        rep = max(5, 50 // max(len(cx), 1))
        cx  = np.repeat(cx, rep, axis=0)
        cy  = np.repeat(cy, rep, axis=0)
        sx  = np.concatenate([sx, cx])
        sy  = np.concatenate([sy, cy])
        print(f"      Corrections added: {len(cx)}  (x{rep} repeats)")
    else:
        print("      No corrections found")

    # ── [3] Load invalid samples ─────────────────────────
    print("\n[3/4] Loading invalid samples...")
    x_inv, y_inv = load_invalid_samples()

    # Hold out 500 invalid for validation
    val_inv_n = min(500, len(x_inv))
    x_val = np.concatenate([x_te,          x_inv[:val_inv_n]])
    y_val = np.concatenate([y_te,          y_inv[:val_inv_n]])
    sx    = np.concatenate([sx,            x_inv[val_inv_n:]])
    sy    = np.concatenate([sy,            y_inv[val_inv_n:]])

    # Shuffle
    idx = np.random.permutation(len(sx))
    sx  = sx[idx]
    sy  = sy[idx].astype(np.int32)

    print(f"\n      Training   : {len(sx)}")
    print(f"      Validation : {len(x_val)}")
    print("\n      Samples per class:")
    for c in range(11):
        n     = int(np.sum(sy == c))
        label = f"Digit {c}" if c < 10 else "Invalid"
        print(f"        class {c:>2}  {label:>9} : {n}")

    # ── Correct class weights ────────────────────────────
    cw = make_class_weights(sy)

    # ── [4] Load or build model ──────────────────────────
    print("\n[4/4] Loading model...")
    if MODEL_PATH.exists():
        print("      Fine-tuning existing model...")
        model  = tf.keras.models.load_model(str(MODEL_PATH))
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 5

    elif BASE_PATH.exists():
        print("      Loading base model...")
        model  = tf.keras.models.load_model(str(BASE_PATH))
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 8

    else:
        print("      Building new model from scratch...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ], name="mnist_model")
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        )
        epochs = 15

    # ── Train ────────────────────────────────────────────
    print(f"\n      Training {epochs} epoch(s)...\n")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 3,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 2,
            verbose  = 1,
        ),
    ]

    model.fit(
        sx, sy,
        epochs          = epochs,
        batch_size      = 128,
        validation_data = (x_val, y_val),
        class_weight    = cw,
        callbacks       = callbacks,
        verbose         = 2,
    )

    # ── Evaluate ─────────────────────────────────────────
    print("\n  Per-class accuracy on validation set:")
    preds  = np.argmax(model.predict(x_val, verbose=0), axis=1)
    all_ok = True
    for cls in range(11):
        idx_cls = np.where(y_val == cls)[0]
        if len(idx_cls) == 0:
            continue
        cls_acc = np.mean(preds[idx_cls] == cls) * 100
        label   = f"Digit {cls}" if cls < 10 else "Invalid(A-Z)"
        flag    = "" if cls_acc >= 90 else "  WARNING LOW"
        if cls_acc < 90:
            all_ok = False
        print(f"    {label:>12} : {cls_acc:5.1f}%  ({len(idx_cls)} samples){flag}")

    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n  MNIST Digit Accuracy : {acc*100:.2f}%")
    print(f"  Loss                 : {loss:.4f}")

    model.save(str(MODEL_PATH))
    status = "COMPLETE  all classes >= 90%" if all_ok else "WARNING  some classes below 90%"
    print(f"\n  {status}")
    print(f"  Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    retrain_model()