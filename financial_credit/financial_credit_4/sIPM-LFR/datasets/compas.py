import os

import pandas as pd
import requests

from datasets.standard_dataset import StandardDataset


class CompasDataset(StandardDataset):
    def __init__(self):
        super(CompasDataset, self).__init__()
        self.name = "compas"
        self.protected_attribute_name = "race"
        self.privileged_classes = ["Caucasian"]

        filedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compas")
        self.download(filedir)
        if not os.path.exists(os.path.join(filedir, "compas_train.csv")):
            # if True:
            df = pd.read_csv(os.path.join(filedir, "compas-scores-two-years.csv"))
            df = self.preprocess(df)

            categorical_features = ["sex", "age_cat", "c_charge_degree"]
            cols = [
                "c_charge_degree",
                "race",
                "age_cat",
                "sex",
                "priors_count",
                "days_b_screening_arrest",
                "decile_score",
                "two_year_recid",
            ]
            df = df[cols].copy()
            df = df.rename(columns={"two_year_recid": "result"})
            df.sample(frac=1, random_state=0)
            self.test = df.tail(df.shape[0] // 10 * 3)
            self.train = df.head(df.shape[0] - self.test.shape[0])
            self.train, self.test = super().process(
                self.train,
                self.test,
                categorical_features=categorical_features,
                features_to_drop=[],
                missing_value=["?"],
                favorable_classes=[1],
                protected_attribute_name=self.protected_attribute_name,
                privileged_classes=self.privileged_classes,
            )
            self.train.sample(frac=1, random_state=0)
            n = self.train.shape[0]
            self.val = self.train.tail(n // 10 * 2)
            self.train = self.train.head(n - self.val.shape[0])
            self.train.to_csv(os.path.join(filedir, "compas_train.csv"), index=None)
            self.val.to_csv(os.path.join(filedir, "compas_val.csv"), index=None)
            self.test.to_csv(os.path.join(filedir, "compas_test.csv"), index=None)
        else:
            self.train = pd.read_csv(
                os.path.join(filedir, "compas_train.csv"), index_col=False
            )
            self.val = pd.read_csv(
                os.path.join(filedir, "compas_val.csv"), index_col=False
            )
            self.test = pd.read_csv(
                os.path.join(filedir, "compas_test.csv"), index_col=False
            )

        columns = self.train.columns.values
        self.fair_variables = [ele for ele in columns if "c_charge_degree" in ele]

    def preprocess(self, df):
        df = df.loc[df["days_b_screening_arrest"] <= 30]
        df = df.loc[df["days_b_screening_arrest"] >= -30]
        df = df.loc[df["is_recid"] != -1]
        df = df.loc[df["c_charge_degree"] != "O"]
        df = df.loc[df["score_text"] != "N/A"]
        return df

    def download(self, filedir):
        if not os.path.exists(os.path.join(filedir, "compas-scores-two-years.csv")):
            print("Downloading COMPAS dataset")
            url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
            r = requests.get(url, allow_redirects=True)
            res = r.content.decode("utf8").replace(" ", "").strip("\n")
            open(os.path.join(filedir, "compas-scores-two-years.csv"), "w").write(res)

            print("Download COMPAS dataset successfully!")


def main():
    CompasDataset()


if __name__ == "__main__":
    main()
