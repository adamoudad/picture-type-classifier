import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from skimage import io

class ImageDataset(Dataset):
    """Image dataset
    """
    def __init__(self, data_path, transform=None):
        """

        """
        self.filenames = list(Path(data_path).glob("**/*.png"))
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image = io.imread(self.filenames[idx])
        image = np.random.normal(size=(4,80,80))
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample

dataset = ImageDataset("./pictures/")


class Rescale(object):
    def __init__(self, output_size):
        """Rescale the sample image to a given size
        """
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
    def __call__(self, sample):
        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(sample, (new_h, new_w))

        landmarks = landmarks * [new_w / w, new_h / h]
        return img

# class RandomCrop(object):
#     """Crop randomly the image in a sample
#     """
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#     def __call__(self, sample):
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size

#         top = np.random.randint(0,h-new_h)
#         left = np.random.randint(0,w-new_w)
#         image = sample[top: top+new_h, ]

