import argparse
import os
import random
import string
import sys
import pandas as pd
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from transformers import EsmForMaskedLM, AutoModel, EsmTokenizer, AutoTokenizer
from utils.process_datasets import DatabaseProcessor
from utils.metric_learning_models import BatchFileDataset, Pre_encoded, FusionDTI
# from bertviz import head_view
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

def get_feature(model, dataloader, args, set_type):
    # Create a subdirectory within input_feature_save_path
    subdirectory = os.path.join(args.input_feature_save_path, args.dataset)
    os.makedirs(subdirectory, exist_ok=True)
    
    batch_files = []
    batch_number = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask, label = batch
            prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask= prot_input_ids.to(args.device), prot_attention_mask.to(args.device),drug_input_ids.to(args.device), drug_attention_mask.to(args.device)
            
            prot_embed, drug_embed = model.encoding(prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask)
            prot_embed = prot_embed.cpu()
            drug_embed = drug_embed.cpu()
            prot_attention_mask = prot_attention_mask.cpu()
            drug_attention_mask = drug_attention_mask.cpu()
            label = label.cpu()
        
            # Save each batch to a separate file in the subdirectory
            batch_file = os.path.join(
                subdirectory,
                f"{args.dataset}_{set_type}_batch_{batch_number}.pt"
            )
            torch.save({
                'prot': prot_embed,
                'drug': drug_embed,
                'prot_mask': prot_attention_mask,
                'drug_mask': drug_attention_mask,
                'y': label
            }, batch_file)
            batch_files.append(batch_file)
            batch_number += 1
    return batch_files

def get_data_loader(file_list, batch_file, shuffle=False, num_workers=4):
    dataset = BatchFileDataset(file_list)
    return DataLoader(dataset, batch_file, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x[0])


def encode_pretrained_feature(args):
    # Define the path to check for existing batch files
    input_feat_path = os.path.join(args.input_feature_save_path, args.dataset)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(input_feat_path):
        os.makedirs(input_feat_path)
        # Check if batch files are already saved
    train_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{args.dataset}_train_batch")])
    valid_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{args.dataset}_valid_batch")])
    test_files = sorted([os.path.join(input_feat_path, f) for f in os.listdir(input_feat_path) if f.startswith(f"{args.dataset}_test_batch")])
    
    if train_files and valid_files and test_files:
        print("Batch files found and will be used.")
    else:
        # prot_tokenizer = BertTokenizer.from_pretrained(args.prot_encoder_path, do_lower_case=False)
        prot_tokenizer = EsmTokenizer.from_pretrained(args.prot_encoder_path)
        print("prot_tokenizer", len(prot_tokenizer))

        # drug_tokenizer = AutoTokenizer.from_pretrained(args.drug_encoder_path, trust_remote_code=True)
        drug_tokenizer = AutoTokenizer.from_pretrained(args.drug_encoder_path)
        print("drug_tokenizer", len(drug_tokenizer))

        prot_model = EsmForMaskedLM.from_pretrained(args.prot_encoder_path)
        # drug_model = AutoModel.from_pretrained(args.drug_encoder_path, deterministic_eval=True, trust_remote_code=True)
        drug_model = AutoModel.from_pretrained(args.drug_encoder_path)
        
        model = Pre_encoded(prot_model, drug_model, args)
        model = model.to(args.device)
        prot_model = model.prot_encoder
        drug_model = model.drug_encoder
            
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
            
            # print("decoded_drug_inputs:", query_encodings2["input_ids"][0])
            # print("decoded_drug_inputs:", drug_tokenizer.decode(query_encodings2["input_ids"][0], skip_special_tokens=True))

            attention_mask1 = query_encodings1["attention_mask"].bool()
            attention_mask2 = query_encodings2["attention_mask"].bool()

            return query_encodings1["input_ids"], attention_mask1, query_encodings2["input_ids"], attention_mask2, scores
            
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

        train_files = get_feature(model, train_dataloader, args, "train")
        valid_files = get_feature(model, valid_dataloader, args, "valid")
        test_files = get_feature(model, test_dataloader, args, "test")

    return train_files, valid_files, test_files
    

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=200, patience=10):
    best_auc = 0
    best_model = None
    epochs_without_improvement = 0  # Initialize counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            prot, drug, prot_mask, drug_mask, label = batch
            prot, drug, prot_mask, drug_mask, label = prot.to(device), drug.to(device), prot_mask.to(device), drug_mask.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(prot, drug, prot_mask, drug_mask)
            loss = criterion(output, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for batch in valid_loader:
                prot, drug, prot_mask, drug_mask, label = batch
                prot, drug, prot_mask, drug_mask, label = prot.to(device), drug.to(device), prot_mask.to(device), drug_mask.to(device), label.to(device)
                output = model(prot, drug, prot_mask, drug_mask)
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(label.cpu().numpy())
            auc = roc_auc_score(actuals, predictions)
            print(f'Epoch {epoch+1}: Validation AUC: {auc:.4f}')
            # Log metrics to wandb
            wandb.log({"epoch": epoch + 1, "loss": total_loss / len(train_loader), "val_auc": auc})

            if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
                # Save the best model
                torch.save(best_model, f'{best_model_dir}/best_model.ckpt')
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
        for batch in test_loader:
            prot, drug, prot_mask, drug_mask, label = batch
            prot, drug = prot.to(device), drug.to(device)
            prot_mask, drug_mask = prot_mask.to(device), drug_mask.to(device)
            output = model(prot, drug, prot_mask, drug_mask)
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(label.cpu().numpy())
    auc = roc_auc_score(actuals, predictions)
    aupr = average_precision_score(actuals, predictions)
    accuracy = accuracy_score(actuals, np.array(predictions) > 0.5)
    print(f'Test AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')
    wandb.log({"Test AUC": auc, "AUPR": aupr, "Accuracy": accuracy})
    
if __name__ == "__main__":
    args = parse_config()
    device = torch.device(args.device)
    print(f"Current device: {args.device}.")
    wandb.init(project="DTI_Prediction_with_Token-level_Fusion", config=args, save_code=True)
    
    wandb.config.update(args)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_dir = (
        f"{args.save_path_prefix}{args.dataset}_{args.fusion}")
    os.makedirs(best_model_dir, exist_ok=True)
    args.save_name = best_model_dir

    model = FusionDTI(446, 768, args).to(device)
    criterion = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-8)
        
    # Load features from the saved batch files
    train_files, valid_files, test_files = encode_pretrained_feature(args)
    train_loader = get_data_loader(train_files, batch_file=1, shuffle=True)
    valid_loader = get_data_loader(valid_files, batch_file=1, shuffle=False)
    test_loader = get_data_loader(test_files, batch_file=1, shuffle=False)
    best_model = train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=500)
    model.load_state_dict(best_model)
    test(model, test_loader, device)
    
    wandb.finish()
