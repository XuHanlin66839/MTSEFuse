import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from F_05251 import FusionNet
import time

class FusionTestDataset(Dataset):
    def __init__(self, ir_dir, vis_dir):
        self.ir_dir, self.vis_dir = ir_dir, vis_dir
        exts = {'.png', '.jpg', '.jpeg'}

        ir_dict  = {os.path.splitext(f)[0]: f
                    for f in os.listdir(ir_dir)
                    if os.path.splitext(f)[1].lower() in exts}
        vis_dict = {os.path.splitext(f)[0]: f
                    for f in os.listdir(vis_dir)
                    if os.path.splitext(f)[1].lower() in exts}

        common_stems = sorted(set(ir_dict) & set(vis_dict))
        assert common_stems, "No matching IR/VIS image pair found!"
        assert len(common_stems) == len(ir_dict) == len(vis_dict), \
            "There are mismatched file names in IR and VIS!"

        self.ir_files  = [ir_dict [s] for s in common_stems]
        self.vis_files = [vis_dict[s] for s in common_stems]
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, idx):
        ir_path = os.path.join(self.ir_dir, self.ir_files[idx])
        ir_img  = Image.open(ir_path).convert('L')

        vis_path = os.path.join(self.vis_dir, self.vis_files[idx])
        vis_rgb  = Image.open(vis_path).convert('RGB')
        y, cb, cr = vis_rgb.convert('YCbCr').split()

        ir_tensor = self.tensor_transform(ir_img)
        y_tensor  = self.tensor_transform(y)

        return ir_tensor, y_tensor, cb, cr, self.vis_files[idx]
def custom_collate_fn(batch):
    return batch[0]
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FusionNet(base_c=16).to(device)
    model.load_state_dict(torch.load("./model/0525_dog_0.41.1_3.pth", map_location=device))
    model.eval()
    test_dataset = FusionTestDataset(
        ir_dir = r"E:\astudy\study\third\3\Dataset\2_MSRS\IR",
        vis_dir = r"E:\astudy\study\third\3\Dataset\2_MSRS\VI_RGB"
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    result_dir = "./result/Ours_road_test"
    os.makedirs(result_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    total_infer_time = 0.0
    with torch.no_grad():
        for sample in test_loader:
            ir_tensor, y_tensor, cb, cr, filename = sample
            ir_tensor = ir_tensor.unsqueeze(0).to(device)
            y_tensor = y_tensor.unsqueeze(0).to(device)
            start_time = time.time()
            fused_tensor = model(ir_tensor, y_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            infer_time = end_time - start_time
            total_infer_time += infer_time
            fused_tensor = fused_tensor.squeeze(0).cpu()
            fused_y = to_pil(fused_tensor).convert("L")
            fused_ycbcr = Image.merge("YCbCr", (fused_y, cb, cr))
            fused_rgb = fused_ycbcr.convert("RGB")
            save_path = os.path.join(result_dir, filename)
            fused_rgb.save(save_path)
            print(f"save_path: {save_path}，infer_time: {infer_time:.4f} S")

    total_images = len(test_dataset)
    print(f"\nTotal_images: {total_images} Pairs")
    print(f"Total_infer_time: {total_infer_time:.4f} S")
    print(f"Average_infer_time: {total_infer_time / total_images:.4f} S")

if __name__ == '__main__':
    main()