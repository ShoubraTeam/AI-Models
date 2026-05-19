# ------------------------------------------------------------------
# Face Verification Class Implementation
# ------------------------------------------------------------------





from src.utils.func import load_arcface_model, load_detector, load_image_cv, expand_bbox, load_obj, print_title
import src.utils.confing as CFG
from PIL.Image import fromarray
import torch.nn.functional as torch_func


class FaceVerifier:
    """
    General Class for Face Verification

    Args:
        margin_factor (float): the margin used to expand the bbox around the face

    """

    ARCFACE_WEIGHTS_PATH = r"D:\Education\College\NN\Project\v2\app\weights/model.pth"
    ARCFACE_THRESHOLDS_PATH = r"D:\Education\College\NN\Project\v2\app\weights/thresholds.json"

    def __init__(self, margin_factor = 0.1, similarity_threshold = 0.35):
        
        self.similarity_threshold = similarity_threshold
        self.margin_factor = margin_factor


        # load models
        self.arcface = self.load_model("arcface")
        self.detector = self.load_model("detector")
        # self.Vit = self.load_model("vit")
    
        # load thresholds
        self.arcface_thresholds = self.load_thresholds("arcface")

    def load_model(self, model_name: str):
        if model_name == "arcface":
            return load_arcface_model(path = FaceVerifier.ARCFACE_WEIGHTS_PATH)
        elif model_name == "detector":
            return load_detector()
    
    def load_thresholds(self, model_name: str):
        if model_name == "arcface":
            return load_obj(path = FaceVerifier.ARCFACE_THRESHOLDS_PATH)

    # ------------------------------------------------------------------------------------------
    def detect(self, img):
        detected = self.detector.predict_jsons(image = img)

        if len(detected) > 1:
            raise ValueError("The image has more than one image.")
        elif len(detected) < 0:
            raise ValueError("The detector detects no images, please upload a higher quality image.")
        
        bbox = detected[0]["bbox"]
        bbox = expand_bbox(bbox = bbox, img_height = img.shape[0], img_width = img.shape[1], margin_factor = self.margin_factor)
        x1, y1, x2, y2 = bbox

        face = img[y1:y2, x1:x2]

        return face
    
    # ------------------------------------------------------------------------------------------
    def verify(self, img1, img2):
        """
        Args:
            img1 (ndarray)
            img2 (ndarray)
        """
        # load imgs
        # img1 = load_image_cv(img1_path)
        # img2 = load_image_cv(img2_path)

        # detect faces
        face1 = self.detect(img1)
        face2 = self.detect(img2)

        # transform img
        face1 = CFG.TRANSFORM(fromarray(face1))
        face2 = CFG.TRANSFORM(fromarray(face2))

        face1 = face1.unsqueeze(0)
        face2 = face2.unsqueeze(0)

        # encoding
        enc1 = self.arcface(face1, inference = True)
        enc2 = self.arcface(face2, inference = True)

        # calc similarity
        similairy = torch_func.cosine_similarity(x1 = enc1, x2 = enc2, dim = 1).item()

        # result
        if similairy > self.similarity_threshold:
            return {
                "verified"  : True,
                "similarity": round(similairy, 3),
                "threshold" : round(self.similarity_threshold, 3)
            }
        else:
            return {
                "verified"  : False,
                "similarity": round(similairy, 3),
                "threshold" : round(self.similarity_threshold, 3)
            }
