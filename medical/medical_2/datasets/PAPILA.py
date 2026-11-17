import torch
import pickle
import numpy as np
from PIL import Image
import os
import pickle
from datasets.base import BaseDataset


class PAPILA(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            sa_array = self.dataframe["Gender"].values.astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count)
            weight_class_1 = total_count_sa / (2 * class_1_count)
            self.class_weights = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array
        elif self.sens_name == "Age":
            # age < --age_threshold: 0, age >= --age_threshold: 1
            return self.dataframe["Age_binary"].values.astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = self.dataframe["Diagnosis"].values.astype(np.float32)
        self.class_nums = 2

        return Y

    view = "full"      # "full": whole image, "disc": fixed central square around the optic disc
    img_size = 224

    @staticmethod
    def crop_disc(img):
        """Central square of side 0.6 x height, centred at (0.5 w, 0.48 h): PAPILA images are disc-centred."""
        w, h = img.size
        side = int(0.6 * h)
        left, top = (w - side) // 2, int(0.48 * h - side / 2)
        return img.crop((left, top, left + side, top + side))

    def cache_dir(self):
        # tools/cache_resized.py stores pre-resized lossless copies here
        suffix = "_224" if (self.view, self.img_size) == ("full", 224) else f"_{self.view}{self.img_size}"
        return self.path_to_images + suffix

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        cached = os.path.join(self.cache_dir(), os.path.splitext(item["Path"])[0] + ".png")
        if os.path.exists(cached):
            img = Image.open(cached).convert("RGB")
        else:
            img = Image.open(os.path.join(self.path_to_images, item["Path"])).convert("RGB")
            if self.view == "disc":
                img = self.crop_disc(img)

        if self.transform is not None:
            img = self.transform(img)

        return img
