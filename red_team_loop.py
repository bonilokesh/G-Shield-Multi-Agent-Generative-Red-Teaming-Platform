import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def run_adversarial_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    epochs = 5
    
    # 1. Load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

    # 2. Initialize Agents
    blue_team = BlueTeamNet().to(device)
    blue_team.load_state_dict(torch.load("blue_team_baseline.pth", map_location=device))
    
    red_generator = RedGenerator(latent_dim).to(device)
    red_discriminator = RedDiscriminator().to(device)

    # 3. Optimizers & Losses
    adversarial_loss = nn.BCELoss()
    blue_criterion = nn.CrossEntropyLoss()
    
    optimizer_G = optim.Adam(red_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(red_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_B = optim.Adam(blue_team.parameters(), lr=0.0001)

    print("Starting Automated AI vs AI Red-Teaming Loop...")
    
    for epoch in range(epochs):
        for i, (imgs, real_labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            imgs, real_labels = imgs.to(device), real_labels.to(device)

            # Ground truths for GAN validation
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # -------------------------------
            # Phase A: Train Red Team Attacker
            # -------------------------------
            optimizer_G.zero_grad()
            
            # Sample random noise and target target_labels (trying to mimic true digits)
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
            gen_imgs = red_generator(z, gen_labels)

            # Red Team evaluation 1: Did it look realistic enough to pass internal checks?
            validity = red_discriminator(gen_imgs, gen_labels)
            g_loss_realism = adversarial_loss(validity, valid)

            # Red Team evaluation 2: The Attack! Force Blue Team to misclassify or struggle
            blue_preds = blue_team(gen_imgs)
            # The Attacker wants the Blue Team to blindly agree it's the intended label
            g_loss_attack = blue_criterion(blue_preds, gen_labels)

            # Combined Attacker Objective (Be realistic AND confusing)
            total_g_loss = g_loss_realism + 0.5 * g_loss_attack
            total_g_loss.backward()
            optimizer_G.step()

            # Train Red Discriminator (Internal sanity check)
            optimizer_D.zero_grad()
            d_real_loss = adversarial_loss(red_discriminator(imgs, real_labels), valid)
            d_fake_loss = adversarial_loss(red_discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -------------------------------
            # Phase B: Train Blue Team Defender (Adaptation)
            # -------------------------------
            optimizer_B.zero_grad()
            
            # Blue Team tries to correctly label the mutated adversarial images
            blue_patch_preds = blue_team(gen_imgs.detach())
            b_loss = blue_criterion(blue_patch_preds, gen_labels)
            b_loss.backward()
            optimizer_B.step()

        # Step Evaluation
        blue_team.eval()
        with torch.no_grad():
            test_noise = torch.randn(10, latent_dim).to(device)
            test_labels = torch.tensor([0,1,2,3,4,5,6,7,8,9]).to(device)
            fooled_imgs = red_generator(test_noise, test_labels)
            test_preds = blue_team(fooled_imgs).argmax(dim=1)
            success_rate = (test_preds == test_labels).float().mean().item() * 100
            
        print(f"Round [{epoch+1}/{epochs}] | Attacker Loss: {total_g_loss.item():.4f} | Defender Defense Accuracy: {success_rate:.2f}%")
        blue_team.train()

    torch.save(blue_team.state_dict(), "blue_team_hardened.pth")
    torch.save(red_generator.state_dict(), "red_generator_final.pth")
    print("Red-teaming session complete. Hardened models exported.")

if __name__ == "__main__":
    run_adversarial_loop()
