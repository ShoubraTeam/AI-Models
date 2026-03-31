# Imports
import base64
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
from inference import IdentityRecognizer



app = FastAPI(
    title = 'Identity Verification',
    description = "ArcFace iResNet50 Verification with Min-Similarity TTA",
    version = "1.0.0"
)

recognizer = IdentityRecognizer(model_arch = 'arcface')
UPPER_THRESHOLD = 0.3
LOWER_THRESHOLD = 0.2

# request
class ImagePair(BaseModel):
    image1Base64 : str
    image2Base64 : str


# decoding I/P
def decode_base64_to_cv2(base64_str: str):
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_data = base64.b64decode(base64_str)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image_pil)
    except Exception as e:
        raise HTTPException(status_code = 400, detail = f"Invalid Image Data: {str(e)}")



# End Point
@app.post("/v1/verify")
async def verify(request: ImagePair):
    # get images
    img1 = decode_base64_to_cv2(request.image1Base64)
    img2 = decode_base64_to_cv2(request.image2Base64)


    # crop
    cropped = recognizer.real_time_detect_faces(img1, img2)
    if cropped['status'] == 'error':
        return {
            'status'    : 'error',
            'message'   : cropped['message'],
            'verified'  : False,
            'Similarity': None
        }

    face1, face2 = cropped['faces']


    # prepare
    prepared1 = recognizer.prepare_images([face1])
    prepared2 = recognizer.prepare_images([face2])


    # encode
    encodings = recognizer.encode_images([prepared1, prepared2])

    # sim
    sim = recognizer.calc_proximity(encodings)
    sim = round(float(sim.item()), 4)

    # verify
    if sim > UPPER_THRESHOLD:
        return {
            'state'     : 'success',
            'message'   : None,
            'verified'  : True,
            'similarity': sim
        }

    elif sim <= LOWER_THRESHOLD:
        return {
            'state'     : 'success',
            'message'   : None,
            'verified'  : False,
            'similarity': sim
        }

    else:
        return {
            'state'     : 'retry',
            'message'   : "Please, upload higher quality images",
            'verified'  : None,
            'similarity': sim
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 3001)
