# from doctr.models.obj_detection.factory import from_hub
import numpy as np
from PIL import Image
# import torch
# from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor
from transformers import (DetrFeatureExtractor,
                          DetrForObjectDetection,
                          YolosFeatureExtractor,
                          YolosForObjectDetection
                         )
                         
                         
def tidy_predict(self, image: np.ndarray) -> str:
    """Gives the top prediction for the provided image"""
    pillow_image = Image.fromarray(image.to_numpy(), 'RGB')
    inputs = self.feature_extractor(images=pillow_image, return_tensors="pt")
    outputs = self.pretrained_model(**inputs)
    return outputs
    
    
def build_detr_model(model_name: str):
    model = DetrForObjectDetection.from_pretrained(model_name)
    features = DetrFeatureExtractor.from_pretrained(model_name)
    return model, features
    

def build_yolos_model(model_name: str):
    model = YolosForObjectDetection.from_pretrained(model_name)
    features = YolosFeatureExtractor.from_pretrained(model_name)
    return model, features
    
    
class facebook_detr_resnet_50:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-50'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class hustvl_yolos_small:
    def __init__(self):
        self.model_name = 'hustvl/yolos-small'
        self.pretrained_model, self.feature_extractor = build_yolos_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class detr_doc_table_detection:
    def __init__(self):
        self.model_name = 'TahaDouaji/detr-doc-table-detection'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class facebook_detr_resnet_101:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-101'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class hustvl_yolos_base:
    def __init__(self):
        self.model_name = 'hustvl/yolos-base'
        self.pretrained_model, self.feature_extractor = build_yolos_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class facebook_detr_resnet_50_dc5:
    def __init__(self):
        self.model_name = 'facebook/detr-resnet-50-dc5'
        self.pretrained_model, self.feature_extractor = build_detr_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
class hustvl_yolos_tiny:
    def __init__(self):
        self.model_name = 'hustvl/yolos-tiny'
        self.pretrained_model, self.feature_extractor = build_yolos_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)

    
class hustvl_yolos_small_300:
    def __init__(self):
        self.model_name = 'hustvl/yolos-small-300'
        self.pretrained_model, self.feature_extractor = build_yolos_model(self.model_name)

    def predict(self, image: np.ndarray) -> str:
        return tidy_predict(self, image)
        
        
# class fasterrcnn_mobilenet_v3_large_fpn:
#     def __init__(self):
#         self.pretrained_model = from_hub('mindee/fasterrcnn_mobilenet_v3_large_fpn').eval()
#         print("WARNING: This model requires installation of non-Python dependencies, see https://github.com/mindee/doctr#prerequisites")
# 
#     def predict(self, image: np.ndarray) -> str:
#         pillow_image = Image.fromarray(image.to_numpy(), 'RGB')
#         # Preprocessing
#         transform = Compose([
#             PILToTensor(),
#             ConvertImageDtype(torch.float32),
#         ])
#         input_tensor = transform(pillow_image).unsqueeze(0)
#         # Inference
#         with torch.inference_mode():
#             output = self.pretrained_model(input_tensor)
#         return output