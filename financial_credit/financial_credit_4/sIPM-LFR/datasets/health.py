import os
import pandas as pd

from datasets.standard_dataset import StandardDataset


class HealthDataset(StandardDataset):
    def __init__(self):
        super(HealthDataset, self).__init__()
        self.name = "health"
        self.protected_attribute_name = "age"
        self.privileged_classes = ["None"]
        filedir = "datasets/health/"

        self.train = pd.read_csv(
            os.path.join(filedir, "health_train.csv"), index_col=False
        )
        self.val = pd.read_csv(
            os.path.join(filedir, "health_val.csv"), index_col=False
        )
        self.test = pd.read_csv(
            os.path.join(filedir, "health_test.csv"), index_col=False
        )


def main():
    HealthDataset()


if __name__ == "__main__":
    main()
