import os

import pandas as pd
import requests

from datasets.standard_dataset import StandardDataset


class AdultDataset(StandardDataset):
    def __init__(self):
        super(AdultDataset, self).__init__()
        self.name = "adult"
        self.protected_attribute_name = "sex"
        self.privileged_classes = ["Male"]

        filedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adult")
        self.download(filedir)
        if not os.path.exists(os.path.join(filedir, "adult_train.csv")):
            print("Generating adult train/val/test dataset")
            self.train = pd.read_csv(os.path.join(filedir, "adult.data"), header=None)
            self.test = pd.read_csv(os.path.join(filedir, "adult.test"), header=None)
            columns = [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "native-country",
                "result",
            ]
            self.train.columns = columns
            self.test.columns = columns
            self.train = self.preprocessing(self.train)
            self.test = self.preprocessing(self.test)

            categorical_features = [
                "workclass",
                "education",
                "age",
                "race",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "native-country",
            ]

            self.train, self.test = self.process(
                self.train,
                self.test,
                protected_attribute_name=self.protected_attribute_name,
                privileged_classes=self.privileged_classes,
                missing_value=["?"],
                features_to_drop=["fnlwgt"],
                categorical_features=categorical_features,
                favorable_classes=[">50K", ">50K."],
            )
            self.train.sample(frac=1, random_state=0)
            n = self.train.shape[0]
            self.val = self.train.tail(n // 10 * 2)
            self.train = self.train.head(n - self.val.shape[0])
            self.train.to_csv(os.path.join(filedir, "adult_train.csv"), index=None)
            self.val.to_csv(os.path.join(filedir, "adult_val.csv"), index=None)
            self.test.to_csv(os.path.join(filedir, "adult_test.csv"), index=None)
        else:
            self.train = pd.read_csv(
                os.path.join(filedir, "adult_train.csv"), index_col=False
            )
            self.val = pd.read_csv(
                os.path.join(filedir, "adult_val.csv"), index_col=False
            )
            self.test = pd.read_csv(
                os.path.join(filedir, "adult_test.csv"), index_col=False
            )

        columns = self.train.columns.values
        self.fair_variables = [ele for ele in columns if "occupation" in ele]

    def download(self, filedir):
        if not os.path.exists(os.path.join(filedir, "adult.data")):
            print("Downloading adult income dataset")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            r = requests.get(url, allow_redirects=True)
            res = r.content.decode("utf8").replace(" ", "").strip("\n")
            open(os.path.join(filedir, "adult.data"), "w").write(res)

            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
            r = requests.get(url, allow_redirects=True)
            res = r.content.decode("utf8").replace(" ", "").strip("\n")
            res = "\n".join(res.split("\n")[1:])
            open(os.path.join(filedir, "adult.test"), "w").write(res)

            print("Download adult income dataset successfully!")

    def preprocessing(self, df):
        def group_edu(x):
            if x <= 5:
                return "<6"
            elif x >= 13:
                return ">12"
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return ">=70"
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df["education-num"] = df["education-num"].apply(lambda x: group_edu(x))
        df["education-num"] = df["education-num"].astype("category")

        # Limit age range
        df["age"] = df["age"].apply(lambda x: x // 10 * 10)
        df["age"] = df["age"].apply(lambda x: age_cut(x))

        # Group race
        df["race"] = df["race"].apply(lambda x: group_race(x))

        return df


def main():
    AdultDataset()


if __name__ == "__main__":
    main()
