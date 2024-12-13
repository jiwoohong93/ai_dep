import optuna
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback


def hyperparameter_search(trial, data_train, data_val, tokenizer, compute_metrics, model_init, output_dir):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    weight_decay = weight_decay if weight_decay != 0.0 else 1e-6
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=weight_decay,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=data_train,
        eval_dataset=data_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    eval_results = trainer.evaluate()
    eval_accuracy = eval_results["eval_accuracy"]

    trial.set_user_attr("trainer_state", trainer.state)
    trial.set_user_attr("learning_rate", learning_rate)
    trial.set_user_attr("batch_size", batch_size)

    return eval_accuracy

def run_hp_search(data_train, data_val, tokenizer, compute_metrics, model_init, output_dir):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: hyperparameter_search(trial, data_train, data_val, tokenizer, compute_metrics, model_init, output_dir), n_trials=3)
    
    return study.best_trial
