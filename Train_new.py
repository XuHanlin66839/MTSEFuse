import os, csv, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from F_05251 import FusionNet
from loss import L_Grad, L_Int
from loss_abla import L_GradX

def laplacian_pyramid(img, levels=3):
    high_freqs, current = [], img
    for _ in range(levels):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up   = F.interpolate(down, scale_factor=2, mode='bilinear', align_corners=False)
        high = current - up
        high_freqs.append(high)
        current = down
    return high_freqs, current

class LowFrequencyLoss(nn.Module):
    def __init__(self, levels=3, epsilon=1e-8,
                 normalize=True, low_freq_strategy='max'):
        super().__init__()
        assert low_freq_strategy in ['max', 'sum']
        self.levels, self.epsilon = levels, epsilon
        self.normalize, self.low_freq_strategy = normalize, low_freq_strategy
        self.mse, self.l1 = nn.MSELoss(), nn.L1Loss()

    def forward(self, fused, ir, vis):
        fused_high, fused_low = laplacian_pyramid(fused, levels=2)
        ir_high, ir_low       = laplacian_pyramid(ir,   levels=2)
        vis_high, vis_low     = laplacian_pyramid(vis,  levels=2)

        if self.low_freq_strategy == 'max':
            target_low = torch.max(ir_low, vis_low) + F.relu(ir_low - vis_low)
        else:
            target_low = ir_low + vis_low

        if self.normalize:
            fused_low  = (fused_low  - fused_low.mean()) / (fused_low.std()  + self.epsilon)
            target_low = (target_low - target_low.mean()) / (target_low.std() + self.epsilon)
            return self.l1(fused_low, target_low)
        else:
            return self.mse(fused_low, target_low)

class FusionDataset(Dataset):
    def __init__(self, ir_dir, vis_dir):
        self.ir_files  = sorted([f for f in os.listdir(ir_dir)  if f.endswith('.png')])
        self.vis_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
        assert self.ir_files == self.vis_files, "IR and VIS file names must correspond one-to-one"
        self.ir_dir, self.vis_dir = ir_dir, vis_dir
        self.transform = transforms.ToTensor()

    def __len__(self): return len(self.ir_files)

    def __getitem__(self, idx):
        ir  = Image.open(os.path.join(self.ir_dir,  self.ir_files[idx])).convert('L')
        vis_rgb = Image.open(os.path.join(self.vis_dir, self.vis_files[idx])).convert('RGB')
        y, _, _ = vis_rgb.convert('YCbCr').split()
        return self.transform(ir), self.transform(y)

def main():
    device, lr, epochs, batch_size = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        1e-3, 40, 8
    )

    os.makedirs("./loss", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join("./loss", f"train_loss_{timestamp}.csv")
    loss_file   = open(log_path, "w", newline="")
    loss_writer = csv.writer(loss_file)
    loss_writer.writerow(["epoch", "batch",
                          "loss_total", "loss_grad", "loss_int", "loss_low"])

    model = FusionNet(base_c=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lowfreq_loss_fn = LowFrequencyLoss(levels=3, normalize=False, low_freq_strategy='max')

    train_loader = DataLoader(
        FusionDataset(
            ir_dir = r"E:\astudy\study\shujuji\MSRS\train\ir",
            vis_dir= r"E:\astudy\study\shujuji\MSRS\train\vi"),
        batch_size=batch_size, shuffle=True, num_workers=4
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        model.train()
        for batch_idx, (ir, vis_y) in enumerate(train_loader):
            ir, vis_y = ir.to(device), vis_y.to(device)
            optimizer.zero_grad()
            fused = model(ir, vis_y)
            loss_grad = L_GradX(ir, vis_y, fused,
                                kind='dog',
                                lamb=7,
                                sigma=3,
                                grad='sobel')

            loss_int  = L_Int(vis_y, ir, fused)
            loss_low  = lowfreq_loss_fn(fused, ir, vis_y)
            loss_total = 1.5 * loss_grad + loss_int + 1.2 * loss_low

            loss_total.backward()
            optimizer.step()

            loss_writer.writerow([epoch, batch_idx,
                                  loss_total.item(),
                                  loss_grad.item(),
                                  loss_int.item(),
                                  loss_low.item()])

            total_loss += loss_total.item()
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch:02d}/{epochs} | Batch {batch_idx:04d}/{len(train_loader)} | "
                      f"Total {loss_total.item():.4f} | G {loss_grad.item():.4f} | "
                      f"I {loss_int.item():.4f} | L {loss_low.item():.4f}")

        print(f"Epoch {epoch:02d}  Average Loss: {total_loss/len(train_loader):.4f}")
        epoch_time = time.time() - epoch_start
        print(f"⏱ Epoch {epoch:02d} finished in {epoch_time / 60:.2f} min "
            f"({epoch_time:.1f} s)")

    loss_file.close()
    print(f"\n✔ Loss log saved to: {log_path}")
    os.makedirs("./model", exist_ok=True)
    torch.save(model.state_dict(), "./model/0525_dog_0.33_7.pth")
    print("✔ Model saved to ./model/0525_dog_0.33_7.pth")

if __name__ == "__main__":
    main()