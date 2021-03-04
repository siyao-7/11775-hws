import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Get_CNN():
    def __init__(self, cuda=False, model_name='resnet34', layer='avgpool', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer and layer_output_size: layer and its output size
        """
        # Model
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model_name
        self.model, self.extraction_layer = self._get_model_and_layer(model_name, layer)
        self.model = self.model.to(self.device)
        self.model.eval()
       
        # Transforms 
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_emb(self, img):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :returns: Numpy ndarray
        """
        # Resize and Normalize Image
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        embedding = torch.zeros(1, self.layer_output_size, 1, 1)
        def copy_data(m, i, o):
            embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return embedding.numpy().squeeze()

    def _get_model_and_layer(self, model_name='resnet34', layer='avgpool'):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resent34'
        :param layer: layer as a string for resnet
        :returns: pytorch model, selected layer
        """

        model = getattr(models, model_name)(pretrained=True)
        layer = model._modules.get(layer) # You can choose your own layer
        return model, layer

if __name__ == "__main__":
    ResNet = tGet_CNN(cuda=False, model_name='resnet34', layer='avgpool', layer_output_size=512)
    img = Image.open('test.jpg')
    emb = ResNet.get_emb(img)
    print(emb.shape)

