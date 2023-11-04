import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.utils import save_image

model = models.vgg19(pretrained = True).features

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        self.chosen_features = ['0','5','10','19','28']
        self.model = models.vgg19(pretrained = True).features[:29]

    def forward(self,x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ]
)

content_img = load_image()
style_img = load_image()

generated = torch.randn(content_img.shape,device= device, requires_grad = True)


