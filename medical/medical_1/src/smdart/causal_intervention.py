"""보호 속성에 대한 인과 개입(do-operator) 기반 추론.

모델은 설문 응답 Q와 보호 속성 A(나이 65세 이상 여부, 성별, 수도권 거주 여부)를 함께 입력받아 학습합니다.
추론 시에는 환자의 실제 보호 속성을 사용하지 않고, A를 가능한 모든 값 a로 강제(do(A=a))한 예측을
학습 데이터의 분포 P(A=a)로 가중 평균합니다 (backdoor adjustment).

    p(Q) = sum_a P(A=a) * f(Q, a)

따라서 추론에 필요한 입력은 설문 응답뿐이며, 예측은 환자의 보호 속성 값에 따라 달라지지 않습니다.
"""

import itertools

import numpy as np
import pandas as pd

PROTECTED_FEATURES = ["age>=65", "gender_male", "region_capital"]


def split_survey_and_protected(columns, non_feature_cols):
    """컬럼 목록을 설문 feature와 보호 속성으로 나눕니다."""
    protected = [c for c in PROTECTED_FEATURES if c in columns]
    survey = [c for c in columns if c not in protected and c not in non_feature_cols]
    return survey, protected


def protected_distribution(A: pd.DataFrame):
    """보호 속성의 모든 조합과 학습 데이터에서의 결합분포 P(A=a)를 반환합니다."""
    combinations = np.array(list(itertools.product([0, 1], repeat=A.shape[1])), dtype=int)
    values = A.to_numpy(dtype=int)
    counts = np.array([(values == combo).all(axis=1).sum() for combo in combinations], dtype=float)
    return combinations, counts / counts.sum()


class CausalInterventionClassifier:
    """보호 속성에 개입한 예측을 가중 평균하는 분류기.

    estimator는 [설문 feature + 보호 속성] 순서의 입력으로 학습된 분류기입니다.
    predict / predict_proba 는 설문 feature만 담긴 DataFrame을 받습니다.
    """

    def __init__(self, estimator, survey_features, protected_features, combinations, weights):
        self.estimator = estimator
        self.survey_features = list(survey_features)
        self.protected_features = list(protected_features)
        self.combinations = np.asarray(combinations, dtype=int)
        self.weights = np.asarray(weights, dtype=float)
        self.classes_ = np.array([0, 1])

    def with_estimator(self, estimator):
        """같은 개입 설정으로 다른 분류기를 감싼 객체를 반환합니다."""
        return CausalInterventionClassifier(
            estimator, self.survey_features, self.protected_features, self.combinations, self.weights
        )

    def _intervened_inputs(self, X: pd.DataFrame):
        """do(A=a)를 적용한 입력을 (가중치, DataFrame) 쌍으로 차례로 반환합니다."""
        Q = X[self.survey_features].reset_index(drop=True)
        for combo, weight in zip(self.combinations, self.weights):
            if weight == 0:
                continue
            A_do = pd.DataFrame(np.tile(combo, (len(Q), 1)), columns=self.protected_features)
            yield weight, pd.concat([Q, A_do], axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.zeros(len(X))
        for weight, X_do in self._intervened_inputs(X):
            p1 += weight * self.estimator.predict_proba(X_do)[:, 1]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba_observed(self, X: pd.DataFrame, A: pd.DataFrame) -> np.ndarray:
        """개입 없이 환자의 실제 보호 속성을 그대로 입력한 예측 (비교 실험용)."""
        X_obs = pd.concat(
            [X[self.survey_features].reset_index(drop=True), A[self.protected_features].reset_index(drop=True)],
            axis=1,
        )
        return self.estimator.predict_proba(X_obs)

    def explain(self, tree_model, X: pd.DataFrame):
        """개입한 예측에 대한 설문 문항별 SHAP 값을 계산합니다.

        tree_model은 [설문 feature + 보호 속성]으로 학습된 XGBoost 모델입니다.
        각 do(A=a)에서의 SHAP 값을 같은 가중치로 평균하며, 보호 속성의 기여분은 기준값에 포함합니다.
        """
        import shap

        explainer = shap.TreeExplainer(tree_model)
        n_survey = len(self.survey_features)
        values = np.zeros((len(X), n_survey))
        base_values = np.zeros(len(X))
        for weight, X_do in self._intervened_inputs(X):
            exp = explainer(X_do)
            values += weight * exp.values[:, :n_survey]
            base_values += weight * (np.asarray(exp.base_values).reshape(-1) + exp.values[:, n_survey:].sum(axis=1))
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=X[self.survey_features].to_numpy(),
            feature_names=self.survey_features,
        )
