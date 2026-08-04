import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter

# ==========================================
# KONFIGURASI
# ==========================================

DATASET_PATH = "dataset"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8

SEED = 42

CLASS_NAMES = [
    "belum_matang",
    "matang",
    "terlalu_matang"
]

# ==========================================
# MEMBACA DATASET
# ==========================================

images = []
labels = []

print("===================================")
print("Membaca Dataset...")
print("===================================")

for label, class_name in enumerate(CLASS_NAMES):

    folder = os.path.join(DATASET_PATH, class_name)

    # Ambil hanya file gambar
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name} : {len(files)} gambar")

    for file in files:

        image_path = os.path.join(folder, file)

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image = tf.keras.utils.img_to_array(image)

        images.append(image)

        labels.append(label)

print("-----------------------------------")
print("Dataset berhasil dibaca.")

print(CLASS_NAMES)

print(Counter(labels))

# ==========================================
# KONVERSI KE NUMPY ARRAY
# ==========================================

images = np.array(images, dtype=np.float32)
labels = np.array(labels)

print("Shape Images :", images.shape)
print("Shape Labels :", labels.shape)

# ==========================================
# DISTRIBUSI LABEL
# ==========================================

unique, counts = np.unique(labels, return_counts=True)

print("\nDistribusi Label")

for u, c in zip(unique, counts):
    print(f"{CLASS_NAMES[u]} : {c}")

print("\nLabel Mapping")

for i, cls in enumerate(CLASS_NAMES):
    print(f"{i} = {cls}")

# ==========================================
# NORMALISASI
# ==========================================

images = images / 255.0

print("\nNormalisasi selesai.")

# ==========================================
# SHUFFLE DATASET
# ==========================================

indices = np.arange(len(images))

np.random.seed(SEED)

np.random.shuffle(indices)

images = images[indices]
labels = labels[indices]

print("Shuffle dataset selesai.")

# ==========================================
# TRAIN + TEMP (70%)
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    images,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

# ==========================================
# VALIDATION + TEST (15% + 15%)
# ==========================================

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

# ==========================================
# INFORMASI DATASET
# ==========================================

print("\n===================================")
print("Pembagian Dataset")
print("===================================")

print(f"Training   : {len(X_train)}")
print(f"Validation : {len(X_valid)}")
print(f"Testing    : {len(X_test)}")

print("\nShape Dataset")

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")

print(f"X_valid : {X_valid.shape}")
print(f"y_valid : {y_valid.shape}")

print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")

print("\n===================================")
print("Preprocessing Selesai")
print("===================================")

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# ==========================================
# MEMBANGUN MODEL CNN
# ==========================================

print("\n===================================")
print("Membangun Model CNN...")
print("===================================")

model = tf.keras.Sequential([

    # Input Layer
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,

    # ======================================
    # Convolution Block 1
    # ======================================

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2,2)
    ),

    # ======================================
    # Convolution Block 2
    # ======================================

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2,2)
    ),

    # ======================================
    # Convolution Block 3
    # ======================================

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2,2)
    ),

    # ======================================
    # Fully Connected
    # ======================================

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )

])

print("Model CNN berhasil dibuat.\n")

model.summary()


# ==========================================
# COMPILE MODEL
# ==========================================

print("\n===================================")
print("Compile Model...")
print("===================================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Compile model berhasil.")

# ==========================================
# CALLBACK
# ==========================================

os.makedirs("models", exist_ok=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/model_jeruk.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# ==========================================
# TRAINING MODEL
# ==========================================

print("\n===================================")
print("Training Model...")
print("===================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_valid, y_valid),

    epochs=50,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],

    verbose=1

)

print("\nTraining selesai.")


# ==========================================
# MEMBUAT FOLDER HASIL
# ==========================================

os.makedirs("hasil", exist_ok=True)

# ==========================================
# EVALUASI MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# ==========================================
# PREDIKSI DATA TEST
# ==========================================

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred_classes
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("hasil/confusion_matrix.png")

plt.close()

print("Confusion Matrix berhasil disimpan.")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred_classes,
    target_names=CLASS_NAMES
)

print("\n===================================")
print("Classification Report")
print("===================================")

print(report)

with open(
    "hasil/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print("Classification Report berhasil disimpan.")

# ==========================================
# GRAFIK ACCURACY
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy Model CNN")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("hasil/accuracy.png")

plt.close()

print("Grafik Accuracy berhasil disimpan.")

# ==========================================
# GRAFIK LOSS
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss Model CNN")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("hasil/loss.png")

plt.close()

print("Grafik Loss berhasil disimpan.")