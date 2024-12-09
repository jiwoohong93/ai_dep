import os
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.datasets import Dataset
from src.utils import remove_padding, entity_f1_func, char_f1_func, set_random_seed

def get_num_labels():
    """Get the number of labels from the KLUE NER dataset."""
    datasets = load_dataset("klue", "ner")
    return len(datasets['train'].info.features['ner_tags'].feature.names)

def load_dataset_local(path):
    """Load dataset from local file"""
    return torch.load(path)

def generate_test_dataset(task, tokenizer, data_dir, max_seq_length=512):
    test_path = os.path.join(data_dir, "klue_test.pt")
    
    if os.path.exists(test_path):
        print("Loading datasets from local files...")
        test_data = load_dataset_local(test_path)
    
    else :
        print("Loading datasets from huggingface...")
        datasets = load_dataset("klue", task)
        test_data = datasets['validation']
    
    LABELS = test_data.info.features['ner_tags'].feature.names
    LABEL2ID = {l:i for i,l in enumerate(LABELS)}
    test_dataset = Dataset(test_data, tokenizer, LABEL2ID, max_seq_length=max_seq_length)
    return test_dataset, LABELS

def load_model_and_tokenizer(model_path):
    """Load the saved model and tokenizer from local files"""
    print(f"\nLoading model from: {model_path}")
    
    # Load the full model checkpoint
    checkpoint = torch.load(os.path.join(model_path, "pytorch_model_full.pt"), map_location="cpu")
    
    # Initialize model from config
    model = AutoModelForTokenClassification.from_config(checkpoint['model_config'])
    
    # Reset classifier if needed
    if model.classifier.out_features != checkpoint['num_labels']:
        clf_in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(clf_in_features, checkpoint['num_labels'])
        model.num_labels = checkpoint['num_labels']
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Number of labels: {model.num_labels}")
    
    return model, tokenizer, checkpoint['labels']  # Also return the labels

def ensemble_predict(ensemble_models, loader, device):
    """Run prediction using ensemble of models."""
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble Evaluation"):
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch
            
            # Get predictions from all models for this batch
            batch_logits = []
            for model in ensemble_models:
                model.eval()
                outputs = model(input_ids, attention_mask)
                batch_logits.append(outputs.logits)
            
            # Average logits for this batch
            avg_logits = torch.mean(torch.stack(batch_logits), dim=0)
            preds = avg_logits.argmax(dim=-1)
            
            # Store predictions and labels
            total_preds.extend(preds.cpu().tolist())
            total_labels.extend(labels.cpu().tolist())
    
    return total_preds, total_labels

def predict(model, loader, device):
    """Run prediction for a single model."""
    model.eval()
    total_preds, total_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            
            preds = outputs.logits.argmax(dim=-1)
            total_preds.extend(preds.cpu().tolist())
            total_labels.extend(labels.cpu().tolist())

    return total_preds, total_labels

def ensemble_evaluate(ensemble_models, loader, device, LABELS):
    """Evaluate using ensemble of models."""
    preds, labels = ensemble_predict(ensemble_models, loader, device)
    preds, labels = remove_padding(preds, labels)
    entity_f1 = entity_f1_func(preds, labels, LABELS)
    char_f1 = char_f1_func(preds, labels, LABELS)
    return entity_f1, char_f1

def evaluate_single_seed(seed_path, device="cuda:0", batch_size=32, max_seq_length=512, outside_weight=1.0,
                         target_models=["roberta-large", "roberta-base"]):
    """Evaluate all models for a single seed, including ensemble."""
    klue_dir = seed_path / "klue"
    if not klue_dir.exists():
        return None
    
    # checkpoint_paths = [d for d in klue_dir.iterdir() if d.is_dir()]
    checkpoint_paths = [d for d in klue_dir.iterdir() 
                   if d.is_dir() and any(target_model in d.name.lower() 
                   for target_model in target_models)]
                       
    checkpoint_paths.sort(key=lambda x: get_model_size(x.name))

    # checkpoint_paths.sort(key=lambda x: get_model_size(x.name))
    
    # Load all models and their results
    models = []
    individual_results = {}
    tokenizer = None
    
    print(f"\nEvaluating models in {seed_path.name}")
    print("-" * 80)
    
    for checkpoint_path in checkpoint_paths:
        try:
            model, tokenizer, _ = load_model_and_tokenizer(checkpoint_path)
            model = model.to(device)
            models.append(model)

            # Generate dataset (same for all models in seed)
            test_dataset, LABELS = generate_test_dataset("ner", tokenizer, Path(os.getcwd()).joinpath("data", "klue-ner"), max_seq_length)
            test_loader = test_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
            
            # Evaluate individual model
            preds, labels = predict(model, test_loader, device)
            preds, labels = remove_padding(preds, labels)
            entity_f1 = entity_f1_func(preds, labels, LABELS)
            char_f1 = char_f1_func(preds, labels, LABELS)
            
            individual_results[checkpoint_path.name] = {
                'entity_f1': entity_f1,
                'char_f1': char_f1,
                'status': 'success'
            }
            
            print(f"\nResults for {checkpoint_path.name}:")
            print(f"Entity F1: {entity_f1:.2f}")
            print(f"Char F1: {char_f1:.2f}")
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {str(e)}")
            individual_results[checkpoint_path.name] = {
                'entity_f1': 0.0,
                'char_f1': 0.0,
                'status': f'error: {str(e)}'
            }
    
    # Perform ensemble evaluation if we have multiple successful models
    ensemble_results = None
    if len(models) > 1:
        print("\nPerforming ensemble evaluation...")
        try:
            entity_f1, char_f1 = ensemble_evaluate(models, test_loader, device, LABELS)
            ensemble_results = {
                'entity_f1': entity_f1,
                'char_f1': char_f1,
                'status': 'success'
            }
            print(f"Ensemble Results:")
            print(f"Entity F1: {entity_f1:.2f}")
            print(f"Char F1: {char_f1:.2f}")
        except Exception as e:
            print(f"Error in ensemble evaluation: {str(e)}")
            ensemble_results = {
                'entity_f1': 0.0,
                'char_f1': 0.0,
                'status': f'error: {str(e)}'
            }
    
    return individual_results, ensemble_results

def get_model_size(name):
    """Extract model size from checkpoint name for sorting."""
    # Split by first '_' to get just the model name part
    model_name = name.split('_')[0]
    if 'bert' in model_name:
        return (0, model_name)
    elif 'small' in model_name.lower():
        return (1, model_name)
    elif 'base' in model_name.lower():
        return (2, model_name)
    elif 'large' in model_name.lower():
        return (3, model_name)
    return (4, model_name)

def main():
    parser = argparse.ArgumentParser(description="Evaluate NER checkpoints across seeds with ensemble")
    parser.add_argument("--base_dir", type=str, default="saved",
                       help="Base directory containing seed folders")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_dir)
    seed_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed=")]
    seed_dirs.sort(key=lambda x: int(x.name.split("=")[1]))

    all_results = {}
    best_scores = {
        'individual': {'entity_f1': 0.0, 'char_f1': 0.0},
        'ensemble': {'entity_f1': 0.0, 'char_f1': 0.0}
    }
    best_configs = {
        'individual': {'entity_f1': None, 'char_f1': None},
        'ensemble': {'entity_f1': None, 'char_f1': None}
    }
    
    for seed_dir in seed_dirs:
        individual_results, ensemble_results = evaluate_single_seed(
            seed_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        all_results[seed_dir.name] = {
            'individual': individual_results,
            'ensemble': ensemble_results
        }
    
    
    successful_ensembles = 0
    entity_f1_scores, char_f1_scores = [], []

    # Print final summary
    print("\nFinal Summary:")
    print("=" * 120)
    
    for seed in sorted(all_results.keys()):
        print(f"\n{seed}:")
        print("-" * 120)
        print(f"{'Model':70} | {'Entity F1':10} | {'Char F1':10} | {'Status'}")
        print("-" * 120)
        
        # Individual results
        results = all_results[seed]['individual']
        sorted_results = sorted(results.items(), key=lambda x: get_model_size(x[0]))

        for model_name, metrics in sorted_results:
            status = "OK" if metrics['status'] == 'success' else "Failed"
            print(f"{model_name:70} | {metrics['entity_f1']:10.2f} | {metrics['char_f1']:10.2f} | {status}")
            
            # Track best individual scores
            if metrics['entity_f1'] > best_scores['individual']['entity_f1']:
                best_scores['individual']['entity_f1'] = metrics['entity_f1']
                best_configs['individual']['entity_f1'] = (seed, model_name)
            if metrics['char_f1'] > best_scores['individual']['char_f1']:
                best_scores['individual']['char_f1'] = metrics['char_f1']
                best_configs['individual']['char_f1'] = (seed, model_name)
        
        # Ensemble results
        if all_results[seed]['ensemble']:
            ensemble_metrics = all_results[seed]['ensemble']
            status = "OK" if ensemble_metrics['status'] == 'success' else "Failed"
            print(f"{'ENSEMBLE':70} | {ensemble_metrics['entity_f1']:10.2f} | {ensemble_metrics['char_f1']:10.2f} | {status}")
            
            # Track best ensemble scores
            if ensemble_metrics['entity_f1'] > best_scores['ensemble']['entity_f1']:
                best_scores['ensemble']['entity_f1'] = ensemble_metrics['entity_f1']
                best_configs['ensemble']['entity_f1'] = seed
            if ensemble_metrics['char_f1'] > best_scores['ensemble']['char_f1']:
                best_scores['ensemble']['char_f1'] = ensemble_metrics['char_f1']
                best_configs['ensemble']['char_f1'] = seed
            
            if metrics['status'] == 'success':
                entity_f1_scores.append(ensemble_metrics['entity_f1'])
                char_f1_scores.append(ensemble_metrics['char_f1'])
                successful_ensembles += 1
    
    # Print best results
    print("\nBest Results:")
    print("=" * 120)
    print("Individual Models:")
    print(f"Best Entity F1: {best_scores['individual']['entity_f1']:.2f} (Seed: {best_configs['individual']['entity_f1'][0]}, Model: {best_configs['individual']['entity_f1'][1]})")
    print(f"Best Char F1: {best_scores['individual']['char_f1']:.2f} (Seed: {best_configs['individual']['char_f1'][0]}, Model: {best_configs['individual']['char_f1'][1]})")
    print("\nEnsemble Models:")
    print(f"Best Entity F1: {best_scores['ensemble']['entity_f1']:.2f} (Seed: {best_configs['ensemble']['entity_f1']})")
    print(f"Best Char F1: {best_scores['ensemble']['char_f1']:.2f} (Seed: {best_configs['ensemble']['char_f1']})")

    # Print average performance
    if successful_ensembles > 0:
        import numpy as np
        
        entity_mean = np.mean(entity_f1_scores)
        entity_std = np.std(entity_f1_scores)
        char_mean = np.mean(char_f1_scores)
        char_std = np.std(char_f1_scores)
        
        print("\nAverage Ensemble Performance:")
        print("=" * 120)
        print(f"Entity F1: {entity_mean:.2f} ± {entity_std:.2f}")
        print(f"Char F1: {char_mean:.2f} ± {char_std:.2f}")

if __name__ == "__main__":
    main()