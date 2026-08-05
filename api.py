import os
import uuid
import psycopg2
import numpy as np
import tensorflow as tf

from PIL import Image
from datetime import datetime, date
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor

from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")

DATABASE_URL = os.getenv("DATABASE_URL")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# ====================================================
# KONFIGURASI
# ====================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "belum_matang",
    "matang",
    "terlalu_matang"
]

def get_connection():

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====================================================
# LOAD MODEL
# ====================================================

print("===================================")
print("Loading CNN Model...")
print("===================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")

# ====================================================
# FASTAPI
# ====================================================

app = FastAPI(
    title="Jeruk CNN API",
    version="1.0"
)

# ====================================================
# CORS
# ====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# STATIC FILE
# ====================================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

# ====================================================
# HALAMAN UTAMA
# ====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Jeruk CNN API Berhasil Berjalan"
    }

# ====================================================
# PREDICT
# ====================================================

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    try:

        # --------------------------------------------
        # VALIDASI FILE
        # --------------------------------------------

        if file.content_type not in [
            "image/jpeg",
            "image/png"
        ]:

            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "File harus berupa JPG atau PNG."
                }
            )

        # --------------------------------------------
        # SIMPAN FILE
        # --------------------------------------------

        filename = f"{uuid.uuid4()}.jpg"

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        with open(filepath, "wb") as f:
            f.write(await file.read())

        # --------------------------------------------
        # LOAD GAMBAR
        # --------------------------------------------

        image = Image.open(filepath)

        image = image.convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image = np.array(image)

        image = image.astype(np.float32)

        image = image / 255.0

        image = np.expand_dims(image, axis=0)

        # --------------------------------------------
        # PREDIKSI
        # --------------------------------------------

        prediction = model.predict(
            image,
            verbose=0
        )

        probabilities = prediction[0]

        class_index = int(np.argmax(probabilities))

        confidence = float(probabilities[class_index])

        # --------------------------------------------
        # DETAIL CONFIDENCE
        # --------------------------------------------

        detail = {}

        for i, cls in enumerate(CLASS_NAMES):

            detail[cls] = round(
                float(probabilities[i]) * 100,
                2
            )

        # --------------------------------------------
        # URL GAMBAR
        # --------------------------------------------

        image_url = f"uploads/{filename}"

        # ====================================================
        # SIMPAN KE DATABASE
        # ====================================================

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO hasil_prediksi
            (
                filename,
                image_url,
                prediction,
                confidence,
                belum_matang,
                matang,
                terlalu_matang,
                create_at
            )
            VALUES
            (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                filename,
                image_url,
                CLASS_NAMES[class_index],
                round(confidence * 100, 2),
                detail["belum_matang"],
                detail["matang"],
                detail["terlalu_matang"],
                date.today()
            )
        )

        conn.commit()

        # --------------------------------------------
        # RESPONSE
        # --------------------------------------------

        return JSONResponse(

            status_code=200,

            content={

                "status": "success",

                "filename": filename,

                "image_url": image_url,

                "prediction": CLASS_NAMES[class_index],

                "confidence": round(confidence, 4),

                "confidence_percent": round(
                    confidence * 100,
                    2
                ),

                "detail": detail,

                "create_at": date.today().isoformat()
            }

        )

    except Exception as e:

        return JSONResponse(

            status_code=500,

            content={

                "status": "error",

                "message": str(e)

            }

        )

# ====================================================
# HISTORY
# ====================================================

@app.get("/history")
def history():

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute("""

            SELECT
                id,
                filename,
                image_url,
                prediction,
                confidence,
                create_at
            FROM hasil_prediksi
            ORDER BY create_at DESC

        """)

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "status": "success",
            "total": len(rows),
            "data": rows
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status":"error",
                "message":str(e)
            }
        )

# ====================================================
# DETAIL HISTORY
# ====================================================

@app.get("/history/{id}")
def detail_history(id: int):

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute("""

            SELECT
                id,
                filename,
                image_url,
                prediction,
                confidence,
                belum_matang,
                matang,
                terlalu_matang,
                create_at
            FROM hasil_prediksi
            WHERE id=%s

        """, (id,))

        row = cur.fetchone()

        cur.close()
        conn.close()

        if row is None:

            return JSONResponse(
                status_code=404,
                content={
                    "status":"error",
                    "message":"Data tidak ditemukan"
                }
            )

        return {
            "status":"success",
            "data":row
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status":"error",
                "message":str(e)
            }
        )

# ====================================================
# LATEST RESULT
# ====================================================

@app.get("/latest")
def latest():

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                id,
                filename,
                image_url,
                prediction,
                confidence,
                belum_matang,
                matang,
                terlalu_matang,
                create_at
            FROM hasil_prediksi
            ORDER BY create_at DESC
            LIMIT 1
            """
        )

        row = cur.fetchone()

        cur.close()
        conn.close()

        if row is None:

            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "Belum ada data prediksi."
                }
            )

        return {
            "status": "success",
            "message": "Data prediksi terbaru berhasil diambil.",
            "data": row
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )