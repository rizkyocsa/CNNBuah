import numpy as np
import tensorflow as tf

# ==========================================
# KONFIGURASI
# ==========================================

MODEL_PATH = "models/model_jeruk.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "belum_matang",
    "matang",
    "terlalu_matang"
]

# ==========================================
# LOAD MODEL
# ==========================================

print("Memuat model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")

# ==========================================
# FUNGSI PREDIKSI
# ==========================================

def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    print("\n=================================")
    print("HASIL PREDIKSI")
    print("=================================")

    print("Kelas :", CLASS_NAMES[predicted_index])

    print(f"Confidence : {confidence*100:.2f}%")

    print("\nProbabilitas")

    for i, kelas in enumerate(CLASS_NAMES):

        print(
            f"{kelas:15s}: {prediction[0][i]*100:.2f}%"
        )

    return {
        "kelas": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probability": prediction[0].tolist()
    }

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    image_path = input("Masukkan lokasi gambar : ")

    predict_image(image_path)