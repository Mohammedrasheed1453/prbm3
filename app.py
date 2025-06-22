import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Generator must match training model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    def forward(self, z):
        return self.gen(z).view(-1, 1, 28, 28)

# Load model
G = Generator()
G.load_state_dict(torch.load("mnist_generator.pth", map_location='cpu'))
G.eval()

# Generate images
def generate_images(num=5):
    z = torch.randn(num, 100)
    with torch.no_grad():
        imgs = G(z)
    return imgs

# Streamlit App
st.title("Handwritten Digit Generator (GAN-MNIST)")
digit = st.selectbox("Choose a digit (for now, GAN is not digit-conditional)", list(range(10)))

if st.button("Generate"):
    imgs = generate_images(5)
    grid = make_grid(imgs, nrow=5, normalize=True)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
