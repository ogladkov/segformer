import cv2 as cv
import glob
import numpy as np
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, datadir, transforms):
        self.filepaths = glob.glob(f"{datadir}/**/*.jpg", recursive=True)
        # self.num_classes = len(classes)
        # self.classes = classes
        self.transforms = transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        print(filepath)
        image = cv.imread(filepath)[:, :, ::-1]
        data = self.transforms(image=np.array(image))

        return data['image']