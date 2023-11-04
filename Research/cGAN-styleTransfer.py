import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from Basics.AutoGrad import epoch
from PIL import Image
import matplotlib.pyplot as plt
to_pil_image = ToPILImage()


class UNetGenerator(nn.Module):
    def __init__ (self,in_channels,out_channels):
        super(UNetGenerator,self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=4,stride = 2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
        )

        #decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.RelU(inplace = True),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,out_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh(),
        )
    def forward(self,x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

class ConditionalGAN(nn.Module):
    def __init__(self,generator,discriminator):
        super(ConditionalGAN).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self,input_image,style_image):
        generated_image = self.generator(style_image)

        real_output = self.discriminator(input_image,style_image)
        fake_output = self.discriminator(generated_image,style_image)

        return generated_image,real_output,fake_output


class PhilippinePaintingDataset(Dataset):
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
    pass # to be defined






if __name__ == "__main__":
    # load dito yung dataset
    images = ...# file path example : ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    styles = ...# file path example : ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    dataset = PhilippinePaintingDataset(images,styles)
    dataloader = DataLoader(dataset,batch_size = 100,shuffle = True)

    # set img_channels
    img_channels = 3
    img_size = 64

    # load generator created or pretrained sa case na to ginawa ko
    generator = UNetGenerator(img_channels,img_channels)
    discriminator = Discriminator(img_channels,img_size)

    cgan = ConditionalGAN(generator,discriminator)

    # evaluation
    generator_loss_fn = nn.MSELoss()
    discriminator_loss_fn = nn.BCELoss()

    optimizer_Generator = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_Discriminator = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    for epoch in range(100):
        for i,(input_image,style_image) in enumerate(dataloader):

            generated_image,real_output,fake_output = cgan(input_image,style_image)

            # calculating loss
            generator_loss = generator_loss_fn(fake_output,real_output)
            discriminator_loss = discriminator_loss_fn(fake_output,real_output)

            optimizer_Generator.zero_grad()
            optimizer_Discriminator.zero_grad()

            generator_loss.backward()
            discriminator_loss.backward()

            optimizer_Generator.step()
            optimizer_Discriminator.step()

        print(f"Epoch {epoch}, Loss_G: {generator_loss.item()}, Loss_D: {discriminator_loss.item()}")


