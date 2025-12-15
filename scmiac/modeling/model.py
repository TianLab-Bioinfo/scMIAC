import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 latent_dim, 
                 hidden_dims=[128, 64], 
                 dropout_rate=0.2):
        super(VAEEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                # nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            in_dim = h_dim
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar



class VAEDecoder(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 output_dim, 
                 hidden_dims=[64, 128], 
                 dropout_rate=0.2, 
                 decoder_activation=None):
        super(VAEDecoder, self).__init__()
        
        self.layers = nn.ModuleList()
        in_dim = latent_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                # nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            in_dim = h_dim
                
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        
        self.decoder_activation = decoder_activation

    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer(h)
        x_recon = self.fc_out(h)
        if self.decoder_activation is not None:
            x_recon = self.decoder_activation(x_recon)
        return x_recon






class VAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        hidden_dims=[128, 64], 
        dropout_rate=0.2, 
        conditional=False, 
        num_classes=0
    ):
        super(VAE, self).__init__()
        
        self.conditional = conditional
        self.num_classes = num_classes
        
        # conditional
        encoder_input_dim = input_dim + num_classes if conditional else input_dim
        decoder_input_dim = latent_dim + num_classes if conditional else latent_dim
        
        self.encoder = VAEEncoder(encoder_input_dim, latent_dim, hidden_dims, dropout_rate)
        self.decoder = VAEDecoder(decoder_input_dim, input_dim, hidden_dims, dropout_rate)

    def encode(self, x, labels=None):
        if self.conditional and labels is not None:
            # one-hot
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
            x = torch.cat([x, labels], dim=1)
        
        # encode
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels=None):
        if self.conditional and labels is not None:
            labels = F.one_hot(labels, num_classes=self.num_classes).float()
            z = torch.cat([z, labels], dim=1)
        
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, labels=None):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        
        return x_recon, mu, logvar








