import io
import os

import torch
import timm

from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from torchvision import transforms


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Diabetic Retinopathy Classification",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DIRECTORIES
# ============================================================

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("========================================")
print("Device:", device)
print("========================================")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "swin_tiny_patch4_window7_224"

NUM_CLASSES = 5


class_names = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferate_DR"
]


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "swin_tiny_diabetic_retinopathy.pth"
)


print("Loading model from:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        "\nModel file not found!\n\n"
        f"Expected location:\n{MODEL_PATH}\n\n"
        "Make sure the file exists inside:\n"
        f"{MODELS_DIR}"
    )


# ============================================================
# CREATE SWIN-TINY
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")


try:

    # --------------------------------------------------------
    # weights_only=False is needed if your checkpoint contains
    # additional Python/Numpy objects.
    #
    # Only use this for a checkpoint you trust, such as the
    # model you trained yourself.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

except TypeError:

    # Compatibility with older PyTorch versions

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


# ============================================================
# EXTRACT STATE DICT
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "✓ Found model_state_dict in checkpoint"
        )

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print(
            "✓ Found state_dict in checkpoint"
        )

    else:

        state_dict = checkpoint

        print(
            "✓ Using checkpoint directly as state_dict"
        )

else:

    state_dict = checkpoint


# ============================================================
# REMOVE POSSIBLE "module." PREFIX
# ============================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key.replace(
            "module.",
            "",
            1
        )

    clean_state_dict[key] = value


# ============================================================
# LOAD WEIGHTS
# ============================================================

missing_keys, unexpected_keys = model.load_state_dict(
    clean_state_dict,
    strict=False
)


if missing_keys:

    print(
        "WARNING - Missing keys:",
        len(missing_keys)
    )


if unexpected_keys:

    print(
        "WARNING - Unexpected keys:",
        len(unexpected_keys)
    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)

model.eval()


print("========================================")
print("✓ Swin-Tiny loaded successfully")
print("========================================")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    index_path = os.path.join(
        TEMPLATES_DIR,
        "index.html"
    )

    if not os.path.isfile(index_path):

        return HTMLResponse(
            content=(
                "<h1>index.html not found</h1>"
                "<p>Put index.html inside the "
                "templates folder.</p>"
            ),
            status_code=500
        )

    with open(
        index_path,
        "r",
        encoding="utf-8"
    ) as f:

        html = f.read()

    return HTMLResponse(
        content=html
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {

        "success": True,

        "status": "API is running",

        "model": MODEL_NAME,

        "device": str(device),

        "classes": class_names

    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        print("\n========================================")
        print("PREDICTION REQUEST")
        print("========================================")

        print(
            "Filename:",
            file.filename
        )

        print(
            "Content type:",
            file.content_type
        )


        # ====================================================
        # VALIDATE CONTENT TYPE
        # ====================================================

        allowed_types = {

            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp"

        }


        if (
            file.content_type
            not in allowed_types
        ):

            return JSONResponse(

                status_code=400,

                content={

                    "success": False,

                    "error":
                        "Unsupported image format. "
                        "Please upload JPG, JPEG, PNG or WEBP."

                }

            )


        # ====================================================
        # READ FILE
        # ====================================================

        contents = await file.read()


        if not contents:

            return JSONResponse(

                status_code=400,

                content={

                    "success": False,

                    "error":
                        "Uploaded file is empty."

                }

            )


        print(
            "File size:",
            len(contents),
            "bytes"
        )


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        try:

            image = Image.open(
                io.BytesIO(contents)
            )

            image = image.convert(
                "RGB"
            )

        except Exception as e:

            print(
                "Image error:",
                repr(e)
            )

            return JSONResponse(

                status_code=400,

                content={

                    "success": False,

                    "error":
                        "The uploaded file is not "
                        "a valid image."

                }

            )


        print(
            "Image size:",
            image.size
        )


        # ====================================================
        # TRANSFORM
        # ====================================================

        image_tensor = transform(
            image
        )


        image_tensor = image_tensor.unsqueeze(
            0
        )


        image_tensor = image_tensor.to(
            device
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


        # ====================================================
        # PREDICTED CLASS
        # ====================================================

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


        predicted_index = (
            predicted_class.item()
        )


        confidence_value = (
            confidence.item() * 100
        )


        prediction = class_names[
            predicted_index
        ]


        # ====================================================
        # PROBABILITIES
        # ====================================================

        probabilities_dict = {

            class_names[i]:

                round(
                    probabilities[
                        0,
                        i
                    ].item() * 100,
                    2
                )

            for i in range(
                NUM_CLASSES
            )

        }


        # ====================================================
        # LOG RESULT
        # ====================================================

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{confidence_value:.2f}%"
        )

        print(
            "Probabilities:",
            probabilities_dict
        )

        print("========================================\n")


        # ====================================================
        # RETURN JSON
        # ====================================================

        return {

            "success": True,

            "prediction":
                prediction,

            "confidence":
                round(
                    confidence_value,
                    2
                ),

            "probabilities":
                probabilities_dict

        }


    except Exception as e:

        print("\n========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(
            repr(e)
        )

        print("========================================\n")


        return JSONResponse(

            status_code=500,

            content={

                "success": False,

                "error":
                    str(e)

            }

        )