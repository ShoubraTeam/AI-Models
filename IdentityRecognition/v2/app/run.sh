


# venv
VENV_PATH=D:/Education/College/______GraduationProject/FaceRecognition/GP_FaceRecognition/Scripts/activate
source "$VENV_PATH"

# run the app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload