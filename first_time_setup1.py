import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# ── Folders ─────────────────────────────────────────────
Path(r"C:\STM32_OTA1\model").mkdir(parents=True,           exist_ok=True)
Path(r"C:\STM32_OTA1\corrections").mkdir(parents=True,     exist_ok=True)
Path(r"C:\STM32_OTA1\model\generated").mkdir(parents=True, exist_ok=True)
Path(r"C:\STM32_OTA1\invalid_samples").mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 11
MODEL       = r"C:\STM32_OTA1\model\mnist.keras"
BASE_MODEL  = r"C:\STM32_OTA1\model\mnist_base.keras"
AZ_CSV      = r"C:\Users\HP\Downloads\archive (8)\A_Z Handwritten Data\A_Z Handwritten Data.csv"


MAX_PER_LETTER = 600   # was 2300 — that caused 59,800 invalid vs 6,000/digit

print("=" * 50)
print("  STM32 OTA1 — First Time Setup")
print("=" * 50)

# ── [1/4] Load MNIST ────────────────────────────────────
print("\n[1/4] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")[..., None] / 255.0
x_test  = x_test.astype("float32")[..., None]  / 255.0
y_train = y_train.astype(np.int32)
y_test  = y_test.astype(np.int32)

print(f"      Train : {x_train.shape}  |  Test : {x_test.shape}")

# ── [2/4] Load A-Z letters (class 10 = Invalid) ─────────
print("\n[2/4] Loading A-Z handwritten CSV  (class 10 = invalid)...")
df     = pd.read_csv(AZ_CSV, header=None)
labels = df.iloc[:, 0].values.astype(np.int32)
pixels = df.iloc[:, 1:].values.astype("float32") / 255.0
x_az   = pixels.reshape(-1, 28, 28, 1)

x_bal, y_bal = [], []
for cls in range(26):
    idx = np.where(labels == cls)[0][:MAX_PER_LETTER]
    x_bal.append(x_az[idx])
    y_bal.append(np.full(len(idx), 10, dtype=np.int32))

x_az = np.concatenate(x_bal, axis=0)
y_az = np.concatenate(y_bal, axis=0)
print(f"      Invalid samples : {len(x_az)}  ({MAX_PER_LETTER} x 26 letters)")

# 10% of invalid → validation
idx_az   = np.random.permutation(len(x_az))
split    = int(0.9 * len(x_az))
x_az_tr  = x_az[idx_az[:split]]
y_az_tr  = y_az[idx_az[:split]]
x_az_val = x_az[idx_az[split:]]
y_az_val = y_az[idx_az[split:]]

x_val = np.concatenate([x_test,  x_az_val], axis=0)
y_val = np.concatenate([y_test,  y_az_val], axis=0)
print(f"      Validation : {len(x_test)} digits + {len(x_az_val)} letters"
      f" = {len(x_val)} total")

# ── [3/4] Combine & shuffle ──────────────────────────────
print("\n[3/4] Combining and shuffling training set...")
x_all = np.concatenate([x_train, x_az_tr], axis=0)
y_all = np.concatenate([y_train, y_az_tr], axis=0).astype(np.int32)

idx   = np.random.permutation(len(x_all))
x_all = x_all[idx]
y_all = y_all[idx]
print(f"      Training   : {len(x_all)} samples")
print(f"      Validation : {len(x_val)} samples")

print("\n      Samples per class in training set:")
for c in range(11):
    n     = int(np.sum(y_all == c))
    label = f"Digit {c}" if c < 10 else "Invalid(A-Z)"
    print(f"        class {c:>2}  {label:>12} : {n}")


n_per_digit = int(np.sum(y_all == 0))
n_invalid   = int(np.sum(y_all == 10))
w_invalid   = n_per_digit / n_invalid

cw = {i: 1.0 for i in range(10)}
cw[10] = w_invalid

print(f"\n      Class weights:")
print(f"        Digits 0-9  : 1.0000  ({n_per_digit} each)")
print(f"        Invalid(10) : {w_invalid:.4f}  ({n_invalid} total)")
print(f"        Loss check  -> digit: {n_per_digit*1.0:.0f}  "
      f"invalid: {n_invalid*w_invalid:.0f}  (should match)")

# ── [4/4] Build & Train ──────────────────────────────────
print("\n[4/4] Building model...")
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
model.summary()

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

print("\n      Training...  (5-8 min on i3, please wait)")
print("      Do NOT close this window\n")

model.fit(
    x_all, y_all,
    epochs          = 15,   # was 10 — more epochs for better convergence
    batch_size      = 128,
    validation_data = (x_val, y_val),
    class_weight    = cw,
    callbacks       = callbacks,
    verbose         = 1,
)

# ── Evaluate ─────────────────────────────────────────────
print("\n" + "-" * 50)
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print(f"  Overall Val Accuracy : {acc * 100:.2f}%")
print(f"  Overall Val Loss     : {loss:.4f}")

print("\n  Per-class accuracy on validation set:")
preds  = np.argmax(model.predict(x_val, verbose=0), axis=1)
all_ok = True
for cls in range(11):
    idx_cls = np.where(y_val == cls)[0]
    if len(idx_cls) == 0:
        continue
    cls_acc = np.mean(preds[idx_cls] == cls) * 100
    label   = f"Digit {cls}" if cls < 10 else "Invalid(A-Z)"
    flag    = "" if cls_acc >= 90 else "  *** WARNING LOW ***"
    if cls_acc < 90:
        all_ok = False
    print(f"    {label:>12} : {cls_acc:5.1f}%  ({len(idx_cls)} samples){flag}")

# ── Save ─────────────────────────────────────────────────
model.save(MODEL)
model.save(BASE_MODEL)

print("\n" + "=" * 50)
if all_ok:
    print("  SETUP COMPLETE  —  all classes >= 90%")
else:
    print("  WARNING  —  some classes below 90%")
    print("  If Invalid < 90%: reduce MAX_PER_LETTER to 400")
    print("  If Digits  < 90%: increase MAX_PER_LETTER to 800")
print("=" * 50)
print(f"  mnist.keras       <- used by pipeline")
print(f"  mnist_base.keras  <- backup (never touched)")
print()
print("  Now run in order:")
print("  1.  python auto_pipeline1.py  <- start OTA")
print("=" * 50)