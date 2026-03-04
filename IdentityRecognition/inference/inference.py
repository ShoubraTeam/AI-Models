import torch
import torch.nn.functional as F
import math
import scripts.load_models as load_models
import scripts.utils as utils
import scripts.config as config
from scripts.utils import load_obj
from pathlib import Path
from PIL.Image import fromarray
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



class IdentityRecognizer():
    # models
    IRESNET_ARCFACE_50 = './models/iresnet_arcface_50.pth'
    IRESNET_SIAMESE_22 = './models/iresnet_siamese_22.pth'

    face_detector = load_models.load_retina_detector()
    transforms = [
        config.val_transform, 
        config.val_transform2,
        config.val_transform3,
        config.val_transform4
    ]


    def __init__(self, model_arch: str):
        """
        Initiating a recognizer.
        Args:
            model_arch: str indicates the arch to use. 
                        should be one of [vgg_siamese - resnet_siamese_10 - resnet_siamese_25 - arcface2 - arcface5]
        """

        # model arch
        if model_arch.lower() not in ['siamese', 'arcface']:
            raise ValueError("Model arch is not allowed.")
        
        self.model_arch = model_arch
        self.model = self.get_model()

        # thresholds
        self.err_threshod = self.get_thresholds()[0]
        self.fr_threshod = self.get_thresholds()[1]
        self.acc_threshold = self.get_thresholds()[2]

    # .............................................................................
    def get_model(self):
        if self.model_arch == 'siamese':
            return load_models.load_siamese_resnet(IdentityRecognizer.IRESNET_SIAMESE_22)
        else: 
            return load_models.load_arcface(IdentityRecognizer.IRESNET_ARCFACE_50)
    # .............................................................................
    def get_thresholds(self):
        """
        This function loads the thresholds

        Returns:
            err_threshold: the threshold at which FRR = FAR
            acc_threshold: the threshold that achieved the max acc
        """
        path = Path(__file__).resolve().parent / f'models/thresholds.json'

        thresholds = load_obj(path)
        if self.model_arch == 'siamese':
            return thresholds['iresnet_siamese_22']
        else:
            return thresholds['iresnet_arcface_50']
    # .............................................................................
    def prepare_images(self, images: list):
        """
        Preparing & Batching the given images
        """
        images_pil = [fromarray(img) for img in images]
        all = []

        for transformer in IdentityRecognizer.transforms:
            transformed = [transformer(img) for img in images_pil]
            transformed = torch.stack(transformed, dim = 0)

            if transformed.dim() == 5: # five-crop
                crops = torch.unbind(transformed, dim = 1)
                all.extend(list(crops))
            else:
                all.append(transformed)

        return all
    # .............................................................................
    def encode_images(self, images):
        """ 
        Use the trained model to encode the given images 

        Args:
            model: the trained model (siamese, arcface)
            images: a sequence image batches
            model_arch: str indicates whether the model is siamese or arcface
        """
        # extract data
        all_imgs1 = images[0]
        all_imgs2 = images[1]

        # encode
        encodings = []
        for transformed_imgs1, transformed_imgs2 in zip(all_imgs1, all_imgs2):
            transformed_imgs1 = transformed_imgs1.to(config.DEVICE)
            transformed_imgs2 = transformed_imgs2.to(config.DEVICE)


            with torch.no_grad():
                if 'siamese' in self.model_arch:
                    e1, e2 = self.model(transformed_imgs1, transformed_imgs2)

                else:
                    e1 = self.model(transformed_imgs1, inference = True)
                    e2 = self.model(transformed_imgs2, inference = True)    
                
                encodings.append(
                    (e1, e2)
                )

        return encodings
    # .............................................................................
    def calc_proximity(self, encodings):
        """
        Args:
            encodings: pairs of encodings, Torch Tensors (b, emb)

        Returns:
            prox: Torch Tensor (float)
        """
        prox = []
        if 'siamese' in self.model_arch:
            for pair in encodings:
                prox.append(F.pairwise_distance(pair[0], pair[1]))

        else:
            for pair in encodings:
                prox.append(F.cosine_similarity(pair[0], pair[1]))

        p = torch.min(torch.stack(prox), dim = 0)
        # p = sum(prox) / len(prox)
        # print(torch.min(torch.stack(prox, dim = 0)).shape)
        # print(p[0].shape)
        
        return p[0]
    # .............................................................................
    def verify(self, pairs: list, threshold: str, debug = False):
        """
        Used to verify betweeb pairs of persons

        Args:
            pairs: list of image pairs (image paths)
            threshold: the type of threshold to use (eer | f1 | acc)
            debug: whether to visualize images with prediction or not

        Retuns:
        """

        # load images
        img1_paths = [Path(__file__).resolve().parent / img1_path for img1_path, _ in pairs]
        img2_paths = [Path(__file__).resolve().parent / img2_path for _, img2_path in pairs]
        imgs1 = [utils.load_image_cv(image_path = img1_path) for img1_path in img1_paths]
        imgs2 = [utils.load_image_cv(image_path = img2_path) for img2_path in img2_paths]

        # crop faces
        imgs1 = [utils.crop_face(face_detector = IdentityRecognizer.face_detector, image = img1) for img1 in imgs1]
        imgs2 = [utils.crop_face(face_detector = IdentityRecognizer.face_detector, image = img2) for img2 in imgs2]

        # get clean images (Removing None from Detector)
        clean_imgs1 = []
        clean_imgs2 = []
        for img1, img2 in zip(imgs1, imgs2):
            if img1 is not None and img2 is not None:
                clean_imgs1.append(img1)
                clean_imgs2.append(img2)


        # prepare images
        clean_imgs1 = self.prepare_images(clean_imgs1)
        clean_imgs2 = self.prepare_images(clean_imgs2)
        
        # get encodings            
        encodings = self.encode_images(
            images = (clean_imgs1, clean_imgs2),
        )
        
        # calc proximity
        prox = self.calc_proximity(encodings)
    
        # th 
        if threshold == 'eer': th = self.err_threshod
        elif threshold == 'f1': th = self.fr_threshod
        elif threshold == 'acc': th = self.acc_threshold
        else: raise ValueError("Threshold is invalid")


        # result
        if 'siamese' in self.model_arch:
            is_same = prox < th
            metric_name = 'distance'
        else:
            is_same = prox > th
            metric_name = 'similarity'
        
        result = [
            {
                'label' : 'same' if same else 'different',
                metric_name : p.item(),
                'threshold' : th
            }
            for p, same in zip(prox, is_same)
        ]

        if debug:
            self.plot_predictions(clean_imgs1[0], clean_imgs2[0], result, metric_name)

        return result
    
    # .............................................................................
    def plot_predictions(self, imgs1, imgs2, result, metric):
        ncols = 3
        nrows = math.ceil(imgs1.size(0) / 3)

        fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (22, 48))

        for img1, img2, res, ax in zip(imgs1, imgs2, result, axes.flat):
            # turn images into numpy
            img1 = utils.tensor_to_ndarry(utils.denormalize_img_tensor(img1))
            img2 = utils.tensor_to_ndarry(utils.denormalize_img_tensor(img2))

            # concatenate
            img = np.concatenate((img1, img2), axis = 1) # on width

            # get label
            lab = res['label']
            th = res['threshold']
            score = res[metric]
            tit = f'Pred: {lab.capitalize()} | Th: {th:.2f} | {metric.capitalize()} Score: {score:.2f}'

            # plot
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(tit)
        
        for ax in axes.flat:
            if ax.has_data() == False:
                fig.delaxes(ax)
        
        plt.tight_layout(h_pad = 2)
        plt.show()


    # .............................................................................
    def real_time_detect_faces(self, img1, img2):
        """
        Used to real-time inference 

        Args:
            image1: img1 (ndarry)
            image2: img2 (ndarry)

        Retuns:
    """
        # crop faces
        face1 = utils.crop_face(face_detector = IdentityRecognizer.face_detector, image = image1)
        face2 = utils.crop_face(face_detector = IdentityRecognizer.face_detector, image = image2)

        if face1 is None or face2 is None:
            return {
                'status'  : 'error',
                'message' : 'either of the given images has more than one face or no faces at all.',
            }   
        
        else:
            return {
                'status'  : 'success',
                'faces' : [face1, face2]
            }
        
    # .............................................................................
    def real_time_verify(self, image1, image2):
        """
        Used to real-time inference 

        Args:
            image1: image1 (ndarry)
            image2: image2 (ndarry)

        Retuns:
        """

        

        


        # prepare images
        clean_imgs1 = self.prepare_images(clean_imgs1)
        clean_imgs2 = self.prepare_images(clean_imgs2)
        
        # get encodings            
        encodings = self.encode_images(
            images = (clean_imgs1, clean_imgs2),
        )
        
        # calc proximity
        prox = self.calc_proximity(encodings)
    
        # th 
        if threshold == 'eer': th = self.err_threshod
        elif threshold == 'f1': th = self.fr_threshod
        elif threshold == 'acc': th = self.acc_threshold
        else: raise ValueError("Threshold is invalid")


        # result
        if 'siamese' in self.model_arch:
            is_same = prox < th
            metric_name = 'distance'
        else:
            is_same = prox > th
            metric_name = 'similarity'
        
        result = [
            {
                'label' : 'same' if same else 'different',
                metric_name : p.item(),
                'threshold' : th
            }
            for p, same in zip(prox, is_same)
        ]

        if debug:
            self.plot_predictions(clean_imgs1[0], clean_imgs2[0], result, metric_name)

        return result
    
    # ...............

        
        



# def get_random_image_paths(parent_path):
#     from itertools import combinations
#     parent_path = Path(__file__).resolve().parent / parent_path
#     all_paths = [os.path.join(parent_path, p) for p in os.listdir(parent_path)]
#     pairs = list(combinations(all_paths, 2))
#     return pairs


# if __name__ == "__main__":
    # init recognizer
    # recognizer = IdentityRecognizer('arcface2')

    # # get random image paths
    # pairs = get_random_image_paths('images')

    # # verify
    # result = recognizer.verify(pairs = pairs, debug = True)
    
    # print(result)
