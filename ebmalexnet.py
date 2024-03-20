"""
EBM with AlexNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet

class EBMAlexNet(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(EBMAlexNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        self.alexnet.classifier[-1] = nn.Linear(4096, num_classes)
        self.rbm = nn.Linear(num_classes, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.alexnet(x)

    def energy(self, x, y):
        y_pred = self.forward(x)
        v = y_pred - y
        h = torch.sigmoid(self.rbm(v))
        energy = -torch.sum(torch.matmul(v, self.rbm.weight.t()) + self.rbm.bias, dim=1) - torch.sum(h, dim=1)
        return energy

    def nce_loss(self, x, y, noise_samples):
        y_pred = self.forward(x)
        energy_data = self.energy(x, y)
        energy_noise = torch.mean(torch.exp(-self.energy(x, noise_samples)), dim=0)
        loss = torch.mean(energy_data) + torch.log(energy_noise)
        return loss

    def train(self, x, y, noise_samples, lr=0.01, epochs=10):
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = self.nce_loss(x, y, noise_samples)
            loss.backward()
            optimizer.step()

# # Example usage
# num_classes = 10
# hidden_dim = 128
# model = RBMAlexNet(num_classes, hidden_dim)

# # Training data (assuming x_train and y_train are tensors)
# x_train = torch.randn(100, 3, 224, 224)
# y_train = torch.randint(0, num_classes, (100,))
# y_train_onehot = torch.zeros(100, num_classes)
# y_train_onehot.scatter_(1, y_train.unsqueeze(1), 1)

# # Noise samples for NCE loss
# noise_samples = torch.randn(100, num_classes)

# # Training the model
# model.train(x_train, y_train_onehot, noise_samples, lr=0.01, epochs=10)

# # Testing the model
# x_test = torch.randn(1, 3, 224, 224)
# y_pred = model.forward(x_test)
# _, predicted_class = torch.max(y_pred, 1)
# print("Predicted class:", predicted_class.item())