import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import wandb

class ConditionalRobustFCNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes=10, emb_dim=64, hidden_dim=1024):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, y):
        y_emb = self.label_embed(y)
        input_concated = torch.cat([x, y_emb], dim=1)
        return self.net(input_concated)


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=1024, mask=None):
        super().__init__()
        self.dim = dim
        if mask is None:
            mask = torch.arange(dim) % 2
        self.register_buffer("mask", mask.float())
        self.net = ConditionalRobustFCNN(
            input_dim=dim, 
            output_dim=dim*2, 
            num_classes=10,
            emb_dim=64,
            hidden_dim=hidden_dim
        )

    def forward(self, x, y, reverse=False):
        mask = self.mask
        x_masked = x * mask 
        st = self.net(x_masked, y)
        s, t = st.chunk(2, dim=1)
        s = 2.0 * torch.tanh(s)
        inv_mask = 1 - mask
        if not reverse:
            y_out = x_masked + inv_mask * (x * torch.exp(s) + t)
            log_det = (inv_mask * s).sum(dim=1)
            return y_out, log_det
        else:
            y_out = x_masked + inv_mask * ((x - t) * torch.exp(-s))
            log_det = -(inv_mask * s).sum(dim=1)
            return y_out, log_det


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim, n_coupling_layers=8, hidden_dim=1024):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(n_coupling_layers):
            mask = (torch.arange(dim) % 2).float()
            if i % 2 == 1:
                mask = 1 - mask
            self.layers.append(ConditionalAffineCoupling(dim=dim, hidden_dim=hidden_dim, mask=mask))

    def forward(self, x, y):
        log_det_sum = x.new_zeros(x.size(0))
        h = x
        for layer in self.layers:
            h, log_det = layer(h, y, reverse=False)
            log_det_sum = log_det_sum + log_det
        return h, log_det_sum

    def inverse(self, z, y):
        h = z
        for layer in reversed(self.layers):
            h, _ = layer(h, y, reverse=True)
        return h, 0

    def log_prob(self, x, y):
        z, log_det = self.forward(x, y)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return log_pz + log_det


def dequantize(x):
    return (x + torch.rand_like(x) / 256.0).clamp(0.0, 1.0)

def save_conditional_samples(model, epoch, device, out_dir="samples"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    n_per_class = 10 
    labels = torch.cat([torch.tensor([i]*n_per_class) for i in range(10)]).to(device)
    
    z = torch.randn(labels.size(0), model.dim, device=device)
    with torch.no_grad():
        x_sample, _ = model.inverse(z, labels)
        x_img = x_sample.view(-1, 1, 28, 28).clamp(0, 1)
        
        z_img = z.view(-1, 1, 28, 28)
        utils.save_image(x_img, os.path.join(out_dir, f"epoch_{epoch}_X_result.png"), nrow=n_per_class)
        utils.save_image(z_img, os.path.join(out_dir, f"epoch_{epoch}_Z_noise.png"), nrow=n_per_class, normalize=True)
        
        grid_img = utils.make_grid(x_img, nrow=n_per_class)
        wandb.log({"generated_samples": [wandb.Image(grid_img, caption=f"Epoch {epoch}")]})

def generate_digit(model, digit, device, out_path_x="gen_digit_X.png", out_path_z="gen_digit_Z.png"):
    model.eval()
    print(f"Generate number {digit}...")
    z = torch.randn(1, 28*28, device=device)
    label = torch.tensor([digit], device=device)
    
    with torch.no_grad():
        x, _ = model.inverse(z, label)
        x_img = x.view(1, 1, 28, 28).clamp(0, 1)
        utils.save_image(x_img, out_path_x)

        z_img = z.view(1, 1, 28, 28)
        utils.save_image(z_img, out_path_z, normalize=True)
        
    print(f"Save X at {out_path_x} and Z at {out_path_z}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Training on: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    dim = 28 * 28
    model = ConditionalRealNVP(dim=dim, n_coupling_layers=args.n_coupling, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            x = dequantize(imgs).view(imgs.size(0), -1)

            loss = -model.log_prob(x, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        wandb.log({
            "train_loss": avg_loss,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(avg_loss)
        save_conditional_samples(model, epoch, device, args.sample_dir)

    save_path = os.path.join("models", "conditional_robust_fcnn.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Training Complete!")
    return model
