import subprocess
import sys
import warnings

import numpy as np
import xgboost as xgb


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"XGBoost: {xgb.__version__}")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("nvidia-smi:")
            print(result.stdout)
        else:
            print("nvidia-smi failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found in the container.")

    try:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_estimators=2,
            max_depth=1,
            eval_metric="logloss",
        )
        x = np.random.rand(64, 8)
        y = np.random.randint(0, 2, size=64)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            model.fit(x, y)

        warning_messages = [str(record.message) for record in records]
        bad_gpu_messages = [
            message for message in warning_messages
            if "not compiled with CUDA support" in message
            or "No visible GPU is found" in message
            or "setting device to CPU" in message
        ]
        if bad_gpu_messages:
            print("XGBoost emitted GPU fallback warnings:")
            for message in bad_gpu_messages:
                print(message)
            return 1

        print("XGBoost CUDA check: OK")
        return 0
    except Exception as exc:
        print("XGBoost CUDA check: FAILED")
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
