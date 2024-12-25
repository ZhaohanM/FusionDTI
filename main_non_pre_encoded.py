import argparse
import os
import random
import string
import sys
import pandas as pd
from datetime import datetime

sys.path.append("../")
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support
from transformers import EsmTokenizer, EsmForMaskedLM, AutoModel, AutoTokenizer
from utils.process_datasets import DatabaseProcessor
from utils.metric_learning_models_2 import FusionDTI
# import lightgbm as lgb

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument(
        "--prot_encoder_path",
        type=str,
        default="westlake-repl/SaProt_650M_AF2",
        # westlake-repl/SaProt_650M_PDB
        help="path/name of protein encoder model located",
    )
    parser.add_argument(
        "--drug_encoder_path",
        type=str,
        default="HUBioDataLab/SELFormer",
        # "ibm/MoLFormer-XL-both-10pct"
        help="path/name of SMILE pre-trained language model",
    )
    parser.add_argument(
        "--input_feature_save_path",
        type=str,
        default="dataset/processed_DTI_Token",
        help="path of tokenized training data",
    )
    parser.add_argument(
        "--agg_mode", default="mean_all_tok", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument(
        "--fusion", default="CAN", type=str, help="{CAN|BAN}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--use_pooled", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default="save_model_ckp/",
        help="save the result in which directory",
    )
    parser.add_argument(
        "--save_name", default="fine_tune", type=str, help="the name of the saved file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BindingDB",
        help="Name of the dataset to use (e.g., 'BindingDB', 'Human', 'Biosnap')"
    )
    return parser.parse_args()
    

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=200, patience=20):
    best_auc = 0
    best_model = None
    epochs_without_improvement = 0  # Initialize counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader)):
            prot_input, prot_mask, drug_input, drug_mask, label = batch
            prot_input, prot_mask, drug_input, drug_mask, label = prot_input.to(device), prot_mask.to(device), drug_input.to(device), drug_mask.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(prot_input, prot_mask, drug_input, drug_mask)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for step, batch in tqdm(enumerate(valid_loader)):
                prot_input, prot_mask, drug_input, drug_mask, label = batch
                prot_input, prot_mask, drug_input, drug_mask, label = prot_input.to(device), prot_mask.to(device), drug_input.to(device), drug_mask.to(device), label.to(device)
                output = model(prot_input, prot_mask, drug_input, drug_mask)
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(label.cpu().numpy())
            auc = roc_auc_score(actuals, predictions)
            print(f'Epoch {epoch+1}: Validation AUC: {auc:.4f}')
            # Log metrics to wandb
            wandb.log({"epoch": epoch + 1, "loss": total_loss / len(train_loader), "val_auc": auc})

            if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    return best_model


def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_loader)):
            prot_input, prot_mask, drug_input, drug_mask, label = batch
            prot_input, prot_mask, drug_input, drug_mask, label = prot_input.to(device), prot_mask.to(device), drug_input.to(device), drug_mask.to(device), label.to(device)
            output = model(prot_input, prot_mask, drug_input, drug_mask)
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(label.cpu().numpy())
    auc = roc_auc_score(actuals, predictions)
    aupr = average_precision_score(actuals, predictions)
    accuracy = accuracy_score(actuals, np.array(predictions) > 0.5)
    print(f'Test AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')
    wandb.log({"Test AUC": auc, "AUPR": aupr, "Accuracy": accuracy})
    
def collate_fn_batch_encoding(batch):
        query1, query2, scores = zip(*batch)
            
        query_encodings1 = prot_tokenizer.batch_encode_plus(
            list(query1),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        query_encodings2 = drug_tokenizer.batch_encode_plus(
            list(query2),
            max_length=512,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        scores = torch.tensor(list(scores))

        attention_mask1 = query_encodings1["attention_mask"].bool()
        attention_mask2 = query_encodings2["attention_mask"].bool()

        return query_encodings1["input_ids"], attention_mask1, query_encodings2["input_ids"], attention_mask2, scores

if __name__ == "__main__":
    args = parse_config()
    device = torch.device(args.device)
    print(f"Current device: {args.device}.")
    wandb.init(project="DTI_Prediction_with_Token-level_Fusion", config=args, save_code=True)
    
    wandb.config.update(args)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
    best_model_dir = (
        f"{args.save_path_prefix}{args.save_name}_{timestamp_str}_{random_str}/"
    )
    os.makedirs(best_model_dir)
    args.save_name = best_model_dir
    
    Dataset = DatabaseProcessor(args)
    train_examples = Dataset.get_train_examples()
    valid_examples = Dataset.get_val_examples()
    test_examples = Dataset.get_test_examples()

    train_dataloader = DataLoader(
            train_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
    )
    valid_dataloader = DataLoader(
            valid_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
    )
    test_dataloader = DataLoader(
            test_examples,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_batch_encoding,
    )
    print( f"dataset loaded: train-{len(train_examples)}; valid-{len(valid_examples)}; test-{len(test_examples)}")
    
    # prot_tokenizer = BertTokenizer.from_pretrained(args.prot_encoder_path, do_lower_case=False)
    prot_tokenizer = EsmTokenizer.from_pretrained(args.prot_encoder_path)
    print("prot_tokenizer", len(prot_tokenizer))
    # drug_tokenizer = AutoTokenizer.from_pretrained(args.drug_encoder_path, trust_remote_code=True)
    drug_tokenizer = AutoTokenizer.from_pretrained(args.drug_encoder_path)
    print("drug_tokenizer", len(drug_tokenizer))

    prot_encoder = EsmForMaskedLM.from_pretrained(args.prot_encoder_path).to(args.device)
    # drug_model = AutoModel.from_pretrained(args.drug_encoder_path, deterministic_eval=True, trust_remote_code=True)
    drug_encoder = AutoModel.from_pretrained(args.drug_encoder_path).to(args.device)

    model = FusionDTI(prot_encoder, drug_encoder, 1280, 768, args).to(device)
    criterion = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
    
    # Load features from the saved batch files

    best_model = train(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, num_epochs=500)
    model.load_state_dict(best_model)
    test(model, test_dataloader, device)
    
    wandb.finish()
