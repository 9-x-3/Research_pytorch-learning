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
    def __init__ (self,in_channels,out_channels,num_classes):
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

    def forward(self,real_image,style_image,class_label):
        generated_image = self.generator(style_image,class_label)

        real_output = self.discriminator(real_image,style_image)
        fake_output = self.discriminator(generated_image,style_image)

        return generated_image,real_output,fake_output


class PhilippinePaintingDataset(Dataset):
    def __init__(self,images,styles,class_labels):
        self.images = images
        self.styles = styles
        self.class_labels = class_labels

    def __getitem__(self, index):
        image = self.images[index]
        style = self.styles[index]
        class_label = self.class_labels[index]
        return image,style,class_label

    def __len__(self):
        return len(self.images)

class Discriminator(nn.Module):
    def __init__(self,num_classes ):
        super(Discriminator,self).__init__()

         # dito lalabas yung condition
        self.class_embedding = nn.Embedding(num_classes, 256)
        self.efficientnet_model = models.efficientnet_b0(pretrained=True)
        num_features = self.efficientnet_model._fc.in_features
        self.fc = nn.Linear(self.efficientnet_model.fc.in_features + 256, 1)

    def forward(self,input_image,class_label):
        class_embedding = self.class_embedding(class_label)

        features = self.efficientnet.extract_features(input_image)
        features = features.mean([2,3])

        x = torch.cat([features,class_embedding], dim=1)

        x = self.fc(x)

        return x

def generate_stylized_image(generator,content_image,style_image,class_label,output_path):

    with torch.no_grad():
        stylized_image = generator(style_image,class_label)

    stylized_image = torch.clamp(stylized_image,0,1)

    stylized_image_pil = to_pil_image(stylized_image[0])
    stylized_image_pil.save(output_path)



def main():
    # load dito dataset
    images = ...# file path example : ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    styles = ...# file path example : ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
    class_labels = ...
    num_classes = len(set(class_labels))
    dataset = PhilippinePaintingDataset(images,styles)
    dataloader = DataLoader(dataset,batch_size = 100,shuffle = True)

    # set img_channels
    img_channels = 3
    img_size = 64

    # load generator created or pretrained sa case na to ginawa ko
    generator = UNetGenerator(img_channels,img_channels,num_classes)
    discriminator = Discriminator(num_classes)

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

if __name__ == "__main__":
    main()

    content_image_path = 'content.jpg'
    style_image_path = 'style.jpg'
    class_label = 0
    output_path = 'stylized_image.jpg'

    content_image = Image.open(style_image_path).convert('RGB')
    content_image = ToTensor()(content_image).unsqueeze(0)

    style_image = Image.open(style_image_path).convert('RGB')
    style_image = ToTensor()(style_image).unsqueeze(0)

    generate_stylized_image(UNetGenerator,content_image_path,class_label,output_path)


