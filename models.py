import torch
import torch.nn as nn

# 1. The Blue Team Defender (Standard CNN Classifier)
class BlueTeamNet(nn.Module):
    def __init__(self):
        super(BlueTeamNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 Classes for digits 0-9
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 2. The Red Team Generator (Creates Adversarial Modifications)
class RedGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(RedGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh() # Outputs values between -1 and 1
        )

    def forward(self, noise, labels):
        # Condition the generator on target labels
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

# 3. The Red Team Internal Discriminator (Forces realism)
class RedDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(RedDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        c = self.label_emb(labels)
        x = torch.cat([img_flat, c], 1)
        return self.model(x)
