"""
EBM with AlexNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet

class EBMAlexNet(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, noise_samples):
        super(EBMAlexNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.classifier = nn.Identity()
        self.rbm = nn.Linear(256, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.noise_samples = noise_samples

    def forward(self, image, text):
        # Extract features from AlexNet
        features = self.alexnet(image)

        # Pass features through RBM
        hidden = torch.sigmoid(self.rbm(features))

        # Pass hidden activations through hidden layers
        for layer in self.hidden_layers:
            hidden = torch.relu(layer(hidden))

        # Pass hidden activations through output layer
        output = self.output_layer(hidden)

        # Compute NCE loss
        noise_distribution = torch.ones_like(output) / output.size(-1)
        noise_samples = torch.multinomial(noise_distribution, self.noise_samples, replacement=True)
        noise_logits = output.gather(1, noise_samples)
        true_logits = output.gather(1, text.unsqueeze(1))

        noise_loss = torch.mean(torch.log(1 - torch.sigmoid(noise_logits) + 1e-8))
        true_loss = torch.mean(torch.log(torch.sigmoid(true_logits) + 1e-8))
        loss = -(true_loss + noise_loss)

        return loss

    def generate(self, image, max_length):
        # Extract features from AlexNet
        features = self.alexnet(image)

        # Pass features through RBM
        hidden = torch.sigmoid(self.rbm(features))

        # Pass hidden activations through hidden layers
        for layer in self.hidden_layers:
            hidden = torch.relu(layer(hidden))

        # Generate text
        generated_text = []
        for _ in range(max_length):
            output = self.output_layer(hidden)
            predicted_token = torch.argmax(output, dim=-1)
            generated_text.append(predicted_token)

        return generated_text

    def train_step(self, dataloader, optimizer, device):
        self.train()
        total_loss = 0

        for images, texts in dataloader:
            images = images.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()
            loss = self.forward(images, texts)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

# Example usage
num_classes = 10
hidden_dim = 128
model = EBMAlexNet(num_classes, hidden_dim)

# Training data (assuming x_train and y_train are tensors)
x_train = torch.randn(100, 3, 224, 224)
y_train = torch.randint(0, num_classes, (100,))
y_train_onehot = torch.zeros(100, num_classes)
y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1)

# Noise samples for NCE loss
noise_samples = torch.randn(100, num_classes)

# Training the model
model.train(x_train, y_train_onehot, noise_samples, lr=0.01, epochs=10)

# Testing the model
x_test = torch.randn(1, 3, 224, 224)
y_pred = model.forward(x_test)
_, predicted_class = torch.max(y_pred, 1)
print("Predicted class:", predicted_class.item())