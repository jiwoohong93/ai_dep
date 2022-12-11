import numpy as np
import pandas as pd


class StandardDataset:
    def __init__(self):
        self.protected_attribute_name = ""
        self.privileged_classes = []
        self.fair_variables = []

    def process(
        self,
        train,
        test,
        protected_attribute_name,
        privileged_classes,
        missing_value=[],
        features_to_drop=[],
        categorical_features=[],
        favorable_classes=[],
        normalize=True,
    ):
        cols = [
            x
            for x in train.columns
            if x
            not in (
                features_to_drop
                + [protected_attribute_name]
                + categorical_features
                + ["result"]
            )
        ]

        result = []
        for df in [train, test]:
            # drop nan values
            df = df.replace(missing_value, np.nan)
            df = df.dropna(axis=0)

            # drop useless features
            df = df.drop(columns=features_to_drop)

            # create one-hot encoding of categorical features
            df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")

            # map protected attributes to privileged or unprivileged
            pos = np.logical_or.reduce(
                np.equal.outer(privileged_classes, df[protected_attribute_name].values)
            )
            df.loc[pos, protected_attribute_name] = 1
            df.loc[~pos, protected_attribute_name] = 0
            df[protected_attribute_name] = df[protected_attribute_name].astype(int)

            # set binary labels
            pos = np.logical_or.reduce(
                np.equal.outer(favorable_classes, df["result"].values)
            )
            df.loc[pos, "result"] = 1
            df.loc[~pos, "result"] = 0
            df["result"] = df["result"].astype(int)

            result.append(df)

        # standardize numeric columns
        for col in cols:
            data = result[0][col].tolist()
            mean = np.mean(data)
            std = np.std(data)
            result[0][col] = (result[0][col] - mean) / std
            result[1][col] = (result[1][col] - mean) / std

        train = result[0]
        test = result[1]
        for col in train.columns:
            if col not in test.columns:
                test[col] = 0
        cols = train.columns
        test = test[cols]
        assert all(
            train.columns[i] == test.columns[i] for i in range(len(train.columns))
        )

        return train, test

    def analyze(self, df_old, y=None, log=True):
        df = df_old.copy()
        if y is not None:
            df["y hat"] = (y > 0.5).astype(int)
        s = self.protected_attribute_name
        res = dict()
        n = df.shape[0]
        y1 = df.loc[df["result"] == 1].shape[0] / n
        if "y hat" in df.columns:
            yh1s0 = (
                df.loc[(df[s] == 0) & (df["y hat"] == 1)].shape[0]
                / df.loc[df[s] == 0].shape[0]
            )
            yh1s1 = (
                df.loc[(df[s] == 1) & (df["y hat"] == 1)].shape[0]
                / df.loc[df[s] == 1].shape[0]
            )
            yh1y1s0 = (
                df.loc[(df["y hat"] == 1) & (df["result"] == 1) & (df[s] == 0)].shape[0]
                / df.loc[(df["result"] == 1) & (df[s] == 0)].shape[0]
            )
            yh1y1s1 = (
                df.loc[(df["y hat"] == 1) & (df["result"] == 1) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 1) & (df[s] == 1)].shape[0]
            )
            yh0y0s0 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 0)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 0)].shape[0]
            )
            yh0y0s1 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 1)].shape[0]
            )

            res["acc"] = df.loc[df["result"] == df["y hat"]].shape[0] / n

            res["DP"] = np.abs(yh1s1 - yh1s0)
            tpr = yh1y1s0 - yh1y1s1
            fpr = yh0y0s0 - yh0y0s1
            res["EO"] = np.abs(tpr) * y1 + np.abs(fpr) * (1 - y1)

            fair_variables = self.fair_variables
            count = (
                df.groupby(fair_variables + [s])
                .count()["y hat"]
                .reset_index()
                .rename(columns={"y hat": "count"})
            )
            count_y = (
                df.groupby(fair_variables + [s])
                .sum()["y hat"]
                .reset_index()
                .rename(columns={"y hat": "count_y"})
            )
            count_merge = pd.merge(count, count_y, how="outer", on=fair_variables + [s])
            count_merge["ratio"] = count_merge["count_y"] / count_merge["count"]
            count_merge = count_merge.drop(columns=["count", "count_y"])
            count_merge["ratio"] = (2 * count_merge[s] - 1) * count_merge["ratio"]
            if len(self.fair_variables) > 0:
                result = (
                    count_merge.groupby(fair_variables)
                    .sum()["ratio"]
                    .reset_index(drop=True)
                    .values
                )
            else:
                result = count_merge.sum()["ratio"]

        if len(self.fair_variables) > 0:
            fairs = (
                df.groupby(self.fair_variables).count()[s].reset_index(drop=True).values
            )
            fairs = fairs / np.sum(fairs)
        else:
            fairs = 1
        res["CF"] = np.sum(np.abs(result) * fairs)

        if log:
            for key, value in res.items():
                print(key, "=", value)
        return res
