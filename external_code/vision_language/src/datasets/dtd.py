import os
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader

class DTD:
    def __init__(self, preprocess, location='~/data', batch_size=32, num_workers=16):
        root = os.path.expanduser(location)
        dtd_root = os.path.join(root, 'dtd')
        train_dir = os.path.join(dtd_root, 'train')
        val_dir = os.path.join(dtd_root, 'val')

        # ✅ Ensure data exists
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("📥 DTD data not found — downloading...")
            url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
            download_and_extract_archive(url, download_root=root)
            # After extraction, organize manually if needed.
            # Assume train/val split is created externally or by another script.
            print("✅ Download complete. Make sure the directory structure is correct.")

        # 📁 Load datasets from folders
        self.train_dataset = datasets.ImageFolder(train_dir, transform=preprocess)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

        self.test_dataset = datasets.ImageFolder(val_dir, transform=preprocess)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)

        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]



















# import os
# import torch
# import torchvision.datasets as datasets


# from datasets import load_dataset
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image

# class DTD:
#     def __init__(self,
#                  preprocess,
#                  location=os.path.expanduser('~/data'),
#                  batch_size=32,
#                  num_workers=16):
#         self.dataset = load_dataset("huggingface/dtd", split=split)
#         self.transform = transform
#         self.classes = self.dataset.features["label"].names
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]
#         image = example["image"]
#         label = example["label"]
#         if self.transform:
#             image = self.transform(image)
#         return image, label



# # class DTD:
# #     def __init__(self,
# #                  preprocess,
# #                  location=os.path.expanduser('~/data'),
# #                  batch_size=32,
# #                  num_workers=16):
# #         # Data loading code
# #         traindir = os.path.join(location, 'dtd', 'train')
# #         valdir = os.path.join(location, 'dtd', 'val')

# #         self.train_dataset = datasets.ImageFolder(
# #             traindir, transform=preprocess)
# #         self.train_loader = torch.utils.data.DataLoader(
# #             self.train_dataset,
# #             shuffle=True,
# #             batch_size=batch_size,
# #             num_workers=num_workers,
# #         )

# #         self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
# #         self.test_loader = torch.utils.data.DataLoader(
# #             self.test_dataset,
# #             batch_size=batch_size,
# #             num_workers=num_workers
# #         )
# #         idx_to_class = dict((v, k)
# #                             for k, v in self.train_dataset.class_to_idx.items())
# #         self.classnames = [idx_to_class[i].replace(
# #             '_', ' ') for i in range(len(idx_to_class))]
