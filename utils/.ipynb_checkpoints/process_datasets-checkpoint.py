import json
import sys
import os
import torch
import logging
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

sys.path.append("../")

LOGGER = logging.getLogger(__name__)

class DatabaseProcessor:
    def __init__(self,args, data_dir="/nfs/TF-DTI/dataset/"):
        # Determine the dataset based on args.dataset_name
        if args.dataset == 'BindingDB':
            dataset_file = 'BindingDB.csv'
            self.name = "BindingDB"
        elif args.dataset == 'Human':
            dataset_file = 'Human.csv'
            self.name = "Human"
        elif args.dataset == 'Biosnap':
            dataset_file = 'Biosnap.csv'
            self.name = "Biosnap"
        else:
            raise ValueError("Invalid dataset name provided. Please choose from 'BindingDB', 'Human', or 'Biosnap'.")
        
        
        # Setup dataset directory
        dataset_dir = f"{data_dir}{self.name}/"
        
        # Load datasets
        self.train_dataset_df = pd.read_csv(dataset_dir + "train.csv")
        self.val_dataset_df = pd.read_csv(dataset_dir + "valid.csv")
        self.test_dataset_df = pd.read_csv(dataset_dir + "test.csv")
        
        # Instantiate DTI_Dataset for training, validation, and testing
        self.train_data_loader = DTI_Dataset((
            self.train_dataset_df["Seq"].values,
            self.train_dataset_df["selfies"].values,
            self.train_dataset_df["label"].values,
        ))
        self.val_data_loader = DTI_Dataset((
            self.val_dataset_df["Seq"].values,
            self.val_dataset_df["selfies"].values,
            self.val_dataset_df["label"].values,
        ))
        self.test_data_loader = DTI_Dataset((
            self.test_dataset_df["Seq"].values,
            self.test_dataset_df["selfies"].values,
            self.test_dataset_df["label"].values,
        ))

    def get_train_examples(self, test=False):
        if test == 1:  # Small testing set, to reduce the running time
            return (
                self.train_dataset_df["Seq"].values[:4096],
                self.train_dataset_df["selfies"].values[:4096],
                self.train_dataset_df["label"].values[:4096],
            )
        elif test > 1:
            return (
                self.train_dataset_df["Seq"].values[:test],
                self.train_dataset_df["selfies"].values[:test],
                self.train_dataset_df["label"].values[:test],
            )
        else:
            return self.train_data_loader
    
    def get_val_examples(self, test=False):
        if test == 1:
            return (
                self.val_dataset_df["Seq"].values[:1024],
                self.val_dataset_df["selfies"].values[:1024],
                self.val_dataset_df["label"].values[:1024],
            )
        elif test > 1:
            return (
                self.val_dataset_df["Seq"].values[:test],
                self.val_dataset_df["selfies"].values[:test],
                self.val_dataset_df["label"].values[:test],
            )
        else:
            return self.val_data_loader

    def get_test_examples(self, test=False):
        if test == 1:
            return (
                self.test_dataset_df["Seq"].values[:1024],
                self.test_dataset_df["selfies"].values[:1024],
                self.test_dataset_df["label"].values[:1024],
            )
        elif test > 1:
            return (
                self.test_dataset_df["Seq"].values[:test],
                self.test_dataset_df["selfies"].values[:test],
                self.test_dataset_df["label"].values[:test],
            )
        else:
            return self.test_data_loader

class DTI_Dataset(Dataset):
    """
    Candidate Dataset for:
        ALL drug-target interactions
    """

    def __init__(self, data_examples):
        self.protein_seqs = data_examples[0]
        self.drug_smil = data_examples[1]
        self.scores = data_examples[2]

    def __getitem__(self, query_idx):

        protein_seq = self.protein_seqs[query_idx]
        drug_smi = self.drug_smil[query_idx]
        score = self.scores[query_idx]

        return protein_seq, drug_smi, score

    def __len__(self):
        return len(self.protein_seqs)