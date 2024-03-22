"""
Energy Based Model

For use with NCE to train an energy map from AlexNet features
Simple "neuron" so just a regular relu and linear in and out. Might
expand later and if it is now, forget this.

Going to hard code the shapes just for AlexNet

Inspired From: https://github.com/lifeitech/nce
"""

import torch
import torch.nn as nn
import torch.distributions as D

class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        # The normalizing constant logZ(θ)        
        self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))

        self.f = nn.Sequential(
            nn.Linear(6, 1536),
            nn.ReLU(),
            nn.Linear(1536, 384),
            nn.ReLU(),
            nn.Linear(384, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )

    def forward(self, x):
        log_p = - self.f(x) - self.c
        return log_p
    
    def nce_loss(self, noise, features, gen_noise):
        """
        Noise Contrastive Estimation (NCE) loss function.
        To be used with AlexNet features
        """
        print(f"\nfeatures.shape: {features.shape}")
        logp_x = self.forward(features).squeeze().unsqueeze(0).unsqueeze(1)    # logp(x)
        print(f"logp_x.shape: {logp_x.shape}")
        logq_x = noise.log_prob(features).unsqueeze(1)  # logq(x)
        print(f"logq_x.shape: {logq_x.shape}")
        logp_gen = self.forward(gen_noise).squeeze().unsqueeze(0).unsqueeze(1)  # logp(x̃)
        print(f"logp_gen.shape: {logp_gen.shape}")
        logq_gen = noise.log_prob(gen_noise).unsqueeze(1)  # logq(x̃)
        print(f"logq_gen.shape: {logq_gen.shape}")

        value_data = logp_x - torch.logsumexp(
            torch.cat(
                [logp_x, logq_x],
                dim=0
            ), 
            dim=1, 
            keepdim=True
        )  # log[p(x)/(p(x) + q(x))]
        
        value_gen = logq_gen - torch.logsumexp(
            torch.cat(
                [logp_gen, logq_gen], 
                dim=1
            ), 
            dim=1, 
            keepdim=True
        )  # log[q(x̃)/(p(x̃) + q(x̃))]

        v = value_data.mean() + value_gen.mean()

        r_x = torch.sigmoid(logp_x - logq_x)
        r_gen = torch.sigmoid(logq_gen - logp_gen)

        acc = (
            (r_x > 1/2).sum() + (r_gen > 1/2).sum()
        ).cpu().numpy() / (len(features) + len(gen_noise))

        return -v,  acc
    
    def optimizer(self, lr=0.001):
        return torch.optim.Adam(
            self.parameters(),
            lr=lr
        )