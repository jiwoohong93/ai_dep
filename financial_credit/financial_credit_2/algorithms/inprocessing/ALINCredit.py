import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))

sys.path.append(parent_dir)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # pandas deprecation notices raised inside aif360 loaders

import joblib
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from metrics import common_utils
from sample import aifData
from algorithms.inprocessing.alincredit_function import load_raw, Preprocessor, Ensemble, reweigh, select_threshold

# aifData (AIF360 StandardDataset) does not allow NaN: missing union features are stored as MISSING
# and turned back into NaN for the NaN-native tree ensemble.
MISSING = -9999.0
# Protected value of rows from datasets without a sex attribute: they belong to neither group,
# so they count toward accuracy but not toward group fairness metrics.
NO_SEX = -1.0


class ALINCredit:
    """ALINCredit: union-feature fair credit boosting.

    dataset_name : 'credit_union' (8 public credit datasets in one union feature space),
                   or one of the AIF360 benchmarks 'german', 'adult', 'compas'
    protected    : protected attribute ('sex'; 'race' is also available for adult/compas)
    deo_cap      : DEO x100 cap used when choosing the decision threshold on the validation split

    Baseline (metrics_orig)   : StandardScaler + LogisticRegression, threshold = best balanced accuracy on valid
    Proposed (metrics_transf) : protected attribute removed from the inputs,
                                HistGradientBoosting + LightGBM soft-vote ensemble trained with
                                group-and-label reweighing, one global threshold chosen on valid
                                (best accuracy subject to DEO <= deo_cap)
    """

    def __init__(self, dataset_name: str = 'credit_union', protected: str = 'sex', deo_cap: float = 4.0,
                 seed: int = 1, n_jobs=None, raw=None):
        self.dataset_name = dataset_name
        self.protected = protected
        self.deo_cap = deo_cap
        self.seed = seed
        self.n_jobs = n_jobs
        self.raw = raw
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        dataset_loaders = {
            'adult': load_preproc_data_adult,
            'german': load_preproc_data_german,
            'compas': load_preproc_data_compas
        }

        if self.dataset_name == 'credit_union':
            if self.protected != 'sex':
                raise ValueError("credit_union only supports protected='sex'")
            self.load_credit_union()
        elif self.dataset_name in dataset_loaders:
            self.dataset_orig = dataset_loaders[self.dataset_name]([self.protected])
            self.dataset_orig_train, self.dataset_orig_vt = self.dataset_orig.split([0.7], shuffle=True, seed=self.seed)
            self.dataset_orig_valid, self.dataset_orig_test = self.dataset_orig_vt.split([0.5], shuffle=True, seed=self.seed)
            self.dataset_names = [self.dataset_name]
            self.dataset_ids = {s: np.zeros(len(d.labels), dtype=int) for s, d in
                                [('train', self.dataset_orig_train), ('valid', self.dataset_orig_valid), ('test', self.dataset_orig_test)]}
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

    def load_credit_union(self):
        """Split every credit dataset 70/15/15, fit its preprocessing on its train rows only,
        and stack the splits into three aifData objects over the union feature space."""
        ds, cols = self.raw if self.raw is not None else load_raw()
        self.dataset_names = list(ds)
        frames = {'train': [], 'valid': [], 'test': []}
        ids = {'train': [], 'valid': [], 'test': []}
        for di, (name, (R, kinds, y, sex)) in enumerate(ds.items()):
            perm = np.random.RandomState(self.seed).permutation(len(y))
            n_tr = int(0.7 * len(y))
            n_va = int(0.5 * (len(y) - n_tr))
            idx = {'train': perm[:n_tr], 'valid': perm[n_tr:n_tr + n_va], 'test': perm[n_tr + n_va:]}
            pre = Preprocessor(kinds).fit(R.iloc[idx['train']])
            for split, rows in idx.items():
                df = pre.transform(R.iloc[rows]).reindex(columns=cols).fillna(MISSING).reset_index(drop=True)
                df['sex'] = sex[rows] if sex is not None else NO_SEX
                df['credit'] = y[rows]
                frames[split].append(df)
                ids[split].append(np.full(len(rows), di))

        self.dataset_ids = {s: np.concatenate(v) for s, v in ids.items()}
        for split in ('train', 'valid', 'test'):
            df = pd.concat(frames[split], ignore_index=True)
            no_sex = (df['sex'] == NO_SEX).to_numpy()
            data = aifData(df=df, label_name='credit', favorable_classes=[1],
                           protected_attribute_names=['sex'], privileged_classes=[[1]])
            # StandardDataset maps every non-privileged value to 0; restore "no sex attribute"
            data.protected_attributes[no_sex, 0] = NO_SEX
            data.features[no_sex, data.feature_names.index('sex')] = NO_SEX
            setattr(self, f'dataset_orig_{split}', data)

    # ----- helpers -----
    def labels01(self, dataset):
        return (dataset.labels.ravel() == dataset.favorable_label).astype(int)

    def sex_or_nan(self, dataset):
        s = dataset.protected_attributes[:, dataset.protected_attribute_names.index(self.protected)].astype(float)
        s[s == NO_SEX] = np.nan
        return s

    def baseline_features(self, dataset):
        X = dataset.features.copy()
        X[X == MISSING] = 0.0  # union features are z-scored on train: 0 = train mean
        return X

    def proposed_features(self, dataset):
        keep = [i for i, n in enumerate(dataset.feature_names) if n not in dataset.protected_attribute_names]
        X = dataset.features[:, keep].copy()
        X[X == MISSING] = np.nan
        return X

    def predict(self, dataset, scores, thresh):
        dataset_pred = dataset.copy(deepcopy=True)
        dataset_pred.scores = scores.reshape(-1, 1)
        fav_inds = scores > thresh
        dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
        dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label
        return dataset_pred

    def compute_metrics(self, dataset_pred):
        return common_utils.compute_metrics(self.dataset_orig_test, dataset_pred,
                                            unprivileged_groups=self.unprivileged_groups,
                                            privileged_groups=self.privileged_groups)

    # ----- baseline: StandardScaler + LogisticRegression -----
    def baseline_fit(self, dataset):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(self.baseline_features(dataset))
        y_train = dataset.labels.ravel()
        lmod = LogisticRegression(max_iter=1000)
        lmod.fit(X_train, y_train, sample_weight=dataset.instance_weights)
        return lmod, scaler

    def baseline_scores(self, dataset, model, scaler):
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        return model.predict_proba(scaler.transform(self.baseline_features(dataset)))[:, pos_ind]

    def baseline_threshold(self, scores, y):
        """Threshold with the best balanced accuracy on valid (same grid as the MAF examples)."""
        class_thresh_arr = np.linspace(0.01, 0.99, 100)
        ba_arr = np.zeros(len(class_thresh_arr))
        for idx, class_thresh in enumerate(class_thresh_arr):
            pred = scores > class_thresh
            ba_arr[idx] = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
        return class_thresh_arr[np.argmax(ba_arr)]

    # ----- proposed: union-feature fair boosting -----
    def proposed_fit(self, dataset, fair=True):
        y = self.labels01(dataset)
        w = reweigh(y, self.sex_or_nan(dataset), self.dataset_ids['train']) if fair else None
        return Ensemble(random_state=0, n_jobs=self.n_jobs).fit(self.proposed_features(dataset), y, sample_weight=w)

    def proposed_scores(self, dataset, model):
        return model.predict_proba(self.proposed_features(dataset))[:, 1]

    def summarize(self, dataset_pred, dataset_ref_pred):
        """Per-dataset accuracy, macro accuracy, DEO x100 and policy drop of the proposed model."""
        y = self.labels01(self.dataset_orig_test)
        pf = self.labels01(dataset_pred)
        pr = self.labels01(dataset_ref_pred)
        d = self.dataset_ids['test']
        per_acc = {n: float((pf[d == i] == y[d == i]).mean() * 100) for i, n in enumerate(self.dataset_names)}
        acc_fair = float((pf == y).mean() * 100)
        acc_ref = float((pr == y).mean() * 100)
        sex = self.sex_or_nan(self.dataset_orig_test)
        per_deo = {}
        for i, n in enumerate(self.dataset_names):
            m = (d == i) & ~np.isnan(sex)
            if m.any():
                tpr = [pf[m & (sex == g) & (y == 1)].mean() for g in (1.0, 0.0)]
                per_deo[n] = float(abs(tpr[0] - tpr[1]) * 100)
        return {
            'per_dataset_acc': per_acc,
            'macro_acc': float(np.mean(list(per_acc.values()))),
            'pooled_acc': acc_fair,
            'pooled_acc_unconstrained': acc_ref,
            'deo_x100': float(abs(self.metrics_transf['equal_opportunity_difference']) * 100),
            'per_dataset_deo': per_deo,
            'policy_drop': (acc_ref - acc_fair) / acc_ref * 100,
            'threshold': self.thresh_fair,
        }

    def run(self):
        y_valid = self.labels01(self.dataset_orig_valid)

        # baseline
        model_orig, scaler_orig = self.baseline_fit(self.dataset_orig_train)
        best_class_thresh_orig = self.baseline_threshold(
            self.baseline_scores(self.dataset_orig_valid, model_orig, scaler_orig), y_valid)
        orig_dataset_test_pred = self.predict(self.dataset_orig_test,
            self.baseline_scores(self.dataset_orig_test, model_orig, scaler_orig), best_class_thresh_orig)
        metrics_orig = self.compute_metrics(orig_dataset_test_pred)

        # proposed (reweighed ensemble, DEO-capped threshold)
        model_fair = self.model_fair = self.proposed_fit(self.dataset_orig_train, fair=True)
        self.thresh_fair = select_threshold(self.proposed_scores(self.dataset_orig_valid, model_fair), y_valid,
                                            self.sex_or_nan(self.dataset_orig_valid), self.deo_cap)
        transf_dataset_test_pred = self.predict(self.dataset_orig_test,
            self.proposed_scores(self.dataset_orig_test, model_fair), self.thresh_fair)
        metrics_transf = self.compute_metrics(transf_dataset_test_pred)
        self.metrics_transf = metrics_transf

        # same ensemble without fairness correction, only used for the policy drop
        model_ref = self.model_ref = self.proposed_fit(self.dataset_orig_train, fair=False)
        thresh_ref = self.thresh_ref = select_threshold(self.proposed_scores(self.dataset_orig_valid, model_ref), y_valid)
        ref_dataset_test_pred = self.predict(self.dataset_orig_test,
            self.proposed_scores(self.dataset_orig_test, model_ref), thresh_ref)
        self.summary = self.summarize(transf_dataset_test_pred, ref_dataset_test_pred)

        return metrics_orig, metrics_transf

    # ----- trained model save / inference only -----
    def save_models(self, path):
        """Save the trained proposed model and its reference model (with their thresholds)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({'seed': self.seed, 'deo_cap': self.deo_cap,
                     'model_fair': self.model_fair, 'thresh_fair': self.thresh_fair,
                     'model_ref': self.model_ref, 'thresh_ref': self.thresh_ref}, path, compress=3)

    def run_inference(self, path):
        """Evaluate saved models on the test split without training."""
        ckpt = joblib.load(path)
        if ckpt['seed'] != self.seed:
            raise ValueError(f"checkpoint seed {ckpt['seed']} != seed {self.seed}")
        self.model_fair, self.thresh_fair = ckpt['model_fair'], ckpt['thresh_fair']
        self.model_ref, self.thresh_ref = ckpt['model_ref'], ckpt['thresh_ref']
        transf_dataset_test_pred = self.predict(self.dataset_orig_test,
            self.proposed_scores(self.dataset_orig_test, self.model_fair), self.thresh_fair)
        self.metrics_transf = self.compute_metrics(transf_dataset_test_pred)
        ref_dataset_test_pred = self.predict(self.dataset_orig_test,
            self.proposed_scores(self.dataset_orig_test, self.model_ref), self.thresh_ref)
        self.summary = self.summarize(transf_dataset_test_pred, ref_dataset_test_pred)
        return self.metrics_transf


if __name__ == "__main__":
    model = ALINCredit(dataset_name='credit_union', protected='sex')
    metrics_orig, metrics_transf = model.run()
    print("Metrics for original data:")
    print(metrics_orig)
    print("\nMetrics for transformed data:")
    print(metrics_transf)
    print("\nSummary (proposed model):")
    print(model.summary)
