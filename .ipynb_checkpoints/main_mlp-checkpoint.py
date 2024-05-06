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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve, f1_score, precision_recall_fscore_support
from transformers import EsmTokenizer, EsmForMaskedLM, AutoModel, AutoTokenizer
from utils.process_datasets import DatabaseProcessor
from utils.metric_learning_models import DTI_Metric_Learning, MlPdecoder
# import lightgbm as lgb

wandb.init(project="DTI_Prediction_with_Token_Level_Fusion")


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
    parser.add_argument("--batch_size", type=int, default=128)
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

def get_feature(model, dataloader, args):
    x = list()
    y = list()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask, y1 = batch
            # Prepare inputs as dictionaries
            prot_input = {
                'input_ids': prot_input_ids.to(args.device), 
                'attention_mask': prot_attention_mask.to(args.device)
            }
            drug_input = {
                'input_ids': drug_input_ids.to(args.device), 
                'attention_mask': drug_attention_mask.to(args.device)
            }
            feature_output = model.predict(prot_input, drug_input)
            x1 = feature_output.cpu().numpy()
            x.append(x1)
            y.append(y1.cpu().numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


def encode_pretrained_feature(args):
    input_feat_file = os.path.join(
        args.input_feature_save_path,
        f"{args.dataset}_{args.group_size}_use_{'pooled' if args.use_pooled else 'cls'}_feat.npz",
    )

    if os.path.exists(input_feat_file):
        print(f"load prior feature data from {input_feat_file}.")
        loaded = np.load(input_feat_file)
        x_train, y_train = loaded["x_train"], loaded["y_train"]
        x_valid, y_valid = loaded["x_valid"], loaded["y_valid"]
        x_test, y_test = loaded["x_test"], loaded["y_test"]
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
        
        model = DTI_Metric_Learning(prot_model, drug_model, 446, 768, args)
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

            attention_mask1 = query_encodings1["attention_mask"].bool()
            attention_mask2 = query_encodings2["attention_mask"].bool()

            return query_encodings1["input_ids"], attention_mask1, query_encodings2["input_ids"], attention_mask2, scores
            
        Dataset = DatabaseProcessor(args)
        train_examples = Dataset.get_train_examples()
        # print(f"get training examples: {len(train_examples)}")
        valid_examples = Dataset.get_val_examples()
        # print(f"get validation examples: {len(valid_examples)}")
        test_examples = Dataset.get_test_examples()
        # print(f"get test examples: {len(test_examples)}")

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
        # print( f"dataset loaded: train-{len(train_examples)}; valid-{len(valid_examples)}; test-{len(test_examples)}")

        x_train, y_train = get_feature(model, train_dataloader, args)
        x_valid, y_valid = get_feature(model, valid_dataloader, args)
        x_test, y_test = get_feature(model, test_dataloader, args)

        # Save input feature to reduce encoding time
        np.savez_compressed(
            input_feat_file,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            y_test=y_test,
        )
        # print(f"save input feature into {input_feat_file}")
    return x_train, y_train, x_valid, y_valid, x_test, y_test
    

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=500, patience=10):
    best_auc = 0
    best_model = None
    epochs_without_improvement = 0  # Initialize counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(target.cpu().numpy())
            auc = roc_auc_score(actuals, predictions)
            print(f'Epoch {epoch+1}: Validation AUC: {auc:.4f}')

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
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            predictions.extend(output.squeeze().cpu().numpy())
            actuals.extend(target.cpu().numpy())
    auc = roc_auc_score(actuals, predictions)
    aupr = average_precision_score(actuals, predictions)
    accuracy = accuracy_score(actuals, np.array(predictions) > 0.5)
    print(f'Test AUC: {auc}, AUPR: {aupr}, Accuracy: {accuracy}')

if __name__ == "__main__":
    args = parse_config()
    device = torch.device(args.device)
    print(f"Current device: {args.device}.")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
    best_model_dir = (
        f"{args.save_path_prefix}{args.save_name}_{timestamp_str}_{random_str}/"
    )
    os.makedirs(best_model_dir)
    args.save_name = best_model_dir

    model = MlPdecoder().to(device)
    criterion = nn.BCELoss()
    
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    # optimizer = optim.Adagrad(model.parameters(), lr=1e-2）
    # optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.LBFGS(model.parameters(), lr=1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)  
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-9)
    
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001) 
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    x_train, y_train, x_valid, y_valid, x_test, y_test = encode_pretrained_feature(args)
    
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(torch.Tensor(x_valid), torch.Tensor(y_valid))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model = train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=500)
    model.load_state_dict(best_model)
    test(model, test_loader, device)