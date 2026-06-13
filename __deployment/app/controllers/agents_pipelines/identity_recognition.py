# -----------------------------------------------
# Required workflow for identity_recognition
# -----------------------------------------------
import numpy as np
import cv2

from models.enums import ResponsesEnum
import torchvision.transforms as T
from PIL.Image import fromarray
import torch.nn.functional as torch_funcs
from torch import Tensor
import torch

class IdentityRecognitionPipeline:
    """
    Idenity Recognition Pipeline Class

    Required Methods:
        preprocess(input)        : pre-process the input before calling the agent. If not pre-processing required -> return the input
        call(input)              : invoke/call the agent on the given input
        postprocess(agent_output): post-process the agent output. If no post-processing required -> return the agent_output.
    """
    # ----------------------------------------- Setup ----------------------------------------------
    def __init__(self, 
        agents,
        similarity_threshold: float = 0.35, 
        margin_factor: float = 0.1
    ) -> None:
        """
        Args:
            agents              : the identity recognizer object
            similarity_threshold: the threshold to identify bettwen same / different persons
            margin_factor       : a factor used to expand the bbox of the extracted face
        """
        
        self.detector = agents["detector"]
        self.verifier = agents["verifier"]
        self.card_classifier = agents["card_classifier"]
        self.transform = self.get_transform()
        self.similarity_threshold = similarity_threshold
        self.margin_factor = margin_factor

        self.CARD_LABEL = "a photo containing an identity card"
        self.NOT_CARD_LABEL = "a photo without any card"
        self.card_classification_candidate_labels = [
            self.CARD_LABEL,
            self.NOT_CARD_LABEL,
        ]


    def get_transform(self):
        IMAGE_SIZE = (112, 112)
        MEANs = [0.5, 0.5, 0.5]
        STDs = [0.5, 0.5, 0.5]
        transform = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean = MEANs,
                std = STDs
            )
        ])

        return transform

    # ------------------------------ Pre Processing ------------------------------------
    def _ndarry_to_pil(self, img: np.ndarray):
        return fromarray(img).convert("RGB")

    def is_card(self, img):

        outputs = self.card_classifier(self._ndarry_to_pil(img), candidate_labels = self.card_classification_candidate_labels)
        scores = {item["label"]: float(item["score"]) for item in outputs}

        card_score = scores.get(self.CARD_LABEL, 0.0)
        not_card_score = scores.get(self.NOT_CARD_LABEL, 0.0)

        return card_score > not_card_score
    

    def preprocess(self, input: list[bytes, bytes]):
        # bytes -> ndarry
        try:
            img1 = self.bytes_to_arrays(input[0])
            img2 = self.bytes_to_arrays(input[1])
        except:
            raise

        # is card
        try:
            is_img1_card = self.is_card(img1)
            is_img2_card = self.is_card(img2)
            if is_img1_card == is_img2_card:
                message = (
                    ResponsesEnum.ID_RECO_NO_PERSONAL.value
                    if is_img1_card
                    else ResponsesEnum.ID_RECO_NO_CARD.value
                )

                return {
                    "success": False,
                    "message": message,
                    "faces"  : None
                }
            
        except:
            raise

        # detect faces
        try:
            n_detections1, face1 = self.detect_face(img1)
            n_detections2, face2 = self.detect_face(img2)
        except:
            raise

        if n_detections1 != 1 or n_detections2 != 1:
            return {
                "success": False,
                "message": ResponsesEnum.ID_RECO_ERROR_REQUIRED_HIGH_QUALITY_IMAGE.value,
                "faces"  : None
            }
        
        # transform
        try:
            face1 = self.transform(fromarray(face1)).unsqueeze(0)
            face2 = self.transform(fromarray(face2)).unsqueeze(0)
        except:
            raise
        
        return {
            "success": True,
            "message": None,
            "faces"  : [face1, face2]
        }

    def bytes_to_arrays(self, img: bytes) -> np.ndarray:
        try:
            arr = np.frombuffer(img, np.uint8)
            arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except:
            raise

        return arr
    
    def detect_face(self, img):
        """
        Return number of detections & face detected
        """
        detected = self.detector.predict_jsons(image = img)

        bbox = detected[0]["bbox"]
        bbox = self.expand_bbox(bbox = bbox, img_height = img.shape[0], img_width = img.shape[1])
        x1, y1, x2, y2 = bbox

        face = img[y1:y2, x1:x2]

        return len(detected), face
    
    def expand_bbox(self, bbox, img_height, img_width):
        """
        Expands the bbox by a percentage of its width & height; to ensure the the detected face covers the whole character

        Args:
            bbox: The original bounding box detected 
            img_height, img_width : The img size
            margin_factor: Expansion percentage

        Returns:
            new_bbox: the new bbox after expansion
        """

        # bbox dimensions
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # padding percentage
        padding_x = int(width * self.margin_factor)
        padding_y = int(height * self.margin_factor)

        # expand the bbox
        new_x1 = x1 - padding_x  # to left
        new_y1 = y1 - padding_y
        new_x2 = x2 + padding_x  # to left
        new_y2 = y2 + padding_y

        # ensure the new coordinates are within the image's boundaries 
        new_x1 = max(0, new_x1) # if negative => assign to zero
        new_y1 = max(0, new_y1)
        new_x2 = min(new_x2, img_width) # if > img_width => assign to img_width
        new_y2 = min(new_y2, img_height)

        new_bbox = [
            int(new_x1),
            int(new_y1),
            int(new_x2),
            int(new_y2),
        ]

        return new_bbox
    
    # ------------------------------ Calling Model ------------------------------------
    def call(self, input: list[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Encoding the given tensors & Return their encodings
        """
        # print(input[0].shape)
        try:
            with torch.no_grad():
                enc1 = self.verifier(input[0], inference = True)
                enc2 = self.verifier(input[1], inference = True)
        except:
            raise

        return (enc1, enc2)
    
    # ------------------------------ Post Process ------------------------------------
    def postprocess(self, agent_output: tuple[Tensor, Tensor]) -> dict[str, bool]:
        """
        Verify if the two images belong to the same person

        Results: 
            {
                verified            : Whether they are the same person or not,
                person_embedding    : the embeddings for the persons if verified. None if not verified,
                similarity_threshold: the threshold used to decide same vs different
                similariy_calculated: similarity calculated between the two images
            }
        """

        verification_results = self.verify(agent_output[0], agent_output[1])
        verified = verification_results['verified']
        similarity = verification_results["similarity"]

        if not verified:
            return {
                "verified"            : verified,
                "similarity"          : similarity,
                "similarity_threshold": self.similarity_threshold, 
                "person_embeddings"   : None,
            }
        
        person_embeddings = self.get_person_embeddings(agent_output[0], agent_output[1])
        return {
            "verified"            : verified,
            "similarity"          : similarity,
            "similarity_threshold": self.similarity_threshold, 
            "person_embeddings"   : person_embeddings,
        }


    def verify(self, enc1: Tensor, enc2: Tensor) -> bool:
        """Verify if the two encoings belong to the same person"""
        similarity = torch_funcs.cosine_similarity(x1 = enc1, x2 = enc2, dim = 1).item()

        if similarity >= self.similarity_threshold:
            return {
                "verified"  : True,
                "similarity": round(similarity, 4)
            }
        
        return {
            "verified"  : False,
            "similarity": round(similarity, 4)
        }


    def get_person_embeddings(self, enc1: Tensor, enc2: Tensor) -> list[float]:
        """Average encodings for the two person images"""
        avg_enc = (enc1 + enc2) / 2
        return avg_enc.unsqueeze(0).tolist()