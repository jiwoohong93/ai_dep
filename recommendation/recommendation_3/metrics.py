import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy", trust_remote_code=True)
    f1_metric = load_metric("f1", trust_remote_code=True)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

