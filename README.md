# MTSEFuse #
Official code of "MTSEFuse: Multi-scale Texture Supplementation Enhancement-Guided Infrared and Visible Image Fusion"

The code has published.
### ----To test(test.py)---- ###

Edit the test_dataset path to match your evaluation folder.

Choose the correct model checkpoint:

0525_dog_0.41.1_3.pth — infrared & visible image fusion.

0525_dog_0.81.5_3.pth — medical image fusion.

### ----To Train(Train_new.py)---- ###
Update **ir_dir** and **vis_dir** inside the **train_loader** definition to point to your IR / VIS training images.

Change the final **save_dir** so that weights & logs are stored where you prefer.

✅ Verified on a laptop with an RTX 4060 (8 GB) — full training runs without GPU memory issues.
