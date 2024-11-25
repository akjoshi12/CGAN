import streamlit as st
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import io

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the model
GENERATOR_PATH = "generator 1.pth"

# Model parameters
latent_dim = 100  # Latent dimension
num_classes = 4  # Number of labels (food, drink, inside, outside)
img_shape = (3, 64, 64)  # Shape of generated images

# Define the Generator class
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = torch.nn.Embedding(num_classes, num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            torch.nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        # Concatenate noise and label embeddings
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Load the trained generator model
@st.cache_resource  # Cache the generator to avoid reloading on every interaction
def load_generator():
    generator = Generator(latent_dim, num_classes, img_shape).to(device)
    if os.path.exists(GENERATOR_PATH):
        generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))
        generator.eval()
        return generator
    else:
        st.error("Generator model not found. Please upload a valid generator model.")
        return None

generator = load_generator()

# Streamlit App UI
st.title("GAN Image Generator")
st.write("Generate images using your trained GAN model.")

# Input form for label selection
label = st.selectbox(
    "Select a label for image generation:",
    options=[0, 1, 2, 3],
    format_func=lambda x: ["Food", "Drink", "Inside", "Outside"][x],
)

# Button to generate the image
if st.button("Generate Image"):
    if generator is not None:
        # Generate noise and label tensor
        z = torch.randn(1, latent_dim, device=device)
        labels = torch.tensor([label], device=device)

        # Generate the image
        gen_img = generator(z, labels).detach().cpu()

        # Normalize the image to [0, 1]
        gen_img = (gen_img + 1) / 2  # Convert from [-1, 1] to [0, 1]

        # Convert to PIL Image
        pil_img = transforms.ToPILImage()(gen_img.squeeze(0))

        # Display the generated image
        st.image(pil_img, caption=f"Generated Image for Label: {['Food', 'Drink', 'Inside', 'Outside'][label]}", use_column_width=True)

# Option to reload the model
if st.button("Reload Model"):
    generator = load_generator()
    st.success("Generator model reloaded successfully!")