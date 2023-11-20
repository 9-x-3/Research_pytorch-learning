import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
to_pil_image = ToPILImage()

class UNetGenerator(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNetGenerator,self). __init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.RelU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

class PhilippinePaintingsDataset(Dataset):
    def __init__(self,images,styles):
        self.images = images
        self.styles = styles

    def __getitem__(self, index):
        image = self.images[index]
        style = self.styles[index]
        return image,style
    def __len__(self):
        return len(self.images)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.efficientnet_model = models.efficientnet_b0(pretrained=True)
        num_features = self.efficientnet_model._fc.in_features
        self.fc = nn.Linear(num_features, 1)

    def forward(self, input_image):
        features = self.efficientnet_model.extract_features(input_image)  # Change to use the efficientnet_model
        features = features.mean([2, 3])
        x = self.fc(features)
        return x

def generate_stylized_image(generator,content_image, style_image, output_path):
    with torch.no_grad():
        stylized_image = generator(style_image)

    stylized_image = torch.clamp(stylized_image, 0, 1)

    stylized_image_pil = to_pil_image(stylized_image[0])
    stylized_image_pil.save(output_path)



def main():
    images = ...
    styles = ...
    dataset = PhilippinePaintingsDataset(images,styles)
    dataloader = DataLoader(dataset,batch_size=100,shuffle=True)

    img_channels = 3
    generator = UNetGenerator(img_channels,img_channels)

    generator_loss_fn = nn.MSELoss()
    optimizer_Generator = optim.Adam(generator.parameters(),lr = 0.001,betas = (0.5,0.999))

    for epoch in range(100):
        for i, (input_image, style_image) in enumerate(dataloader):
            generated_image = generator(style_image)

            # Calculating loss
            generator_loss = generator_loss_fn(generated_image, input_image)

            optimizer_Generator.zero_grad()
            generator_loss.backward()
            optimizer_Generator.step()

        print(f"Epoch {epoch}, Loss_G: {generator_loss.item()}")



if __name__ == "__main__":
    main()

    content_image_path = 'content.jpg'
    style_image_path = 'style.jpg'
    output_path = 'stylized_image.jpg'

    content_image = Image.open(content_image_path).convert('RGB')
    content_image = ToTensor()(content_image).unsqueeze(0)

    style_image = Image.open(style_image_path).convert('RGB')
    style_image = ToTensor()(style_image).unsqueeze(0)

    generate_stylized_image(UNetGenerator, content_image, style_image, output_path)
