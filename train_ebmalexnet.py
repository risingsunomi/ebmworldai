"""
Train EBMAlexNet program
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import alexnet

from ebmalexnet import EBMAlexNet
from captures import Captures

def main():
    # Create an instance of the model
    model = EBMAlexNet(vocab_size=10000, hidden_size=512, num_hidden_layers=2, noise_samples=10)

    # Move the model to the appropriate device (e.g., GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Assuming you have your data in the format (image_array, text)
    image_text_data = [...]  # Your list of (image_array, text) tuples

    # Create a cap, text dataset
    caps = Captures()
    dataset = caps.get_all_pframes()

    # Split the dataset into train and validation sets
    train_ratio = 0.8
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for images, texts in train_dataloader:
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()
            loss = model.forward(images, texts)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in val_dataloader:
                images = images.to(device)
                texts = texts.to(device)

                loss = model.forward(images, texts)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")