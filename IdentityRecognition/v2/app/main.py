from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.face_verifier import FaceVerifier
from src.utils.func import bytes_to_numpy, print_title

app = FastAPI(title = "ArcFace Face Verification")
app.mount("/static", StaticFiles(directory = "static"), name = "static")




print_title("Loading Models...")
face_verifier = FaceVerifier()


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/verify")
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    print_title("Performing Verification...")

    # print_title("")

    # get bytes
    image1_bytes = await image1.read()
    image2_bytes = await image2.read()

    # numpy array
    image1_arr = bytes_to_numpy(image1_bytes)
    image2_arr = bytes_to_numpy(image2_bytes)

    print(100 * '-')
    print(image1_arr.shape, type(image1_arr))
    print(100 * '-')

    # verify
    result = face_verifier.verify(
        img1 = image1_arr,
        img2 = image2_arr
    )

    return result