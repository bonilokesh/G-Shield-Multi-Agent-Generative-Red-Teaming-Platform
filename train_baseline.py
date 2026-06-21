import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Baseline Blue Team on device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    blue_team = BlueTeamNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(blue_team.parameters(), lr=0.001)

    # Short 2-epoch training for baseline setup
    blue_team.train()
    for epoch in range(2):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = blue_team(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(blue_team.state_dict(), "blue_team_baseline.pth")
    print("Baseline Blue Team saved successfully as 'blue_team_baseline.pth'.\n")

if __name__ == "__main__":
    train_baseline()
