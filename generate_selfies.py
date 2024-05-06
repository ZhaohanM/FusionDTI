import pandas as pd
import argparse
from utils.prepare_drug import prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--smiles_dataset", default="./dataset/BindingDB.csv", help="Path of the input SMILES dataset.")
parser.add_argument("--selfies_dataset", default="./dataset/BindingDB_SEFLIES.csv", help="Path of the output SEFLIES dataset.")
args = parser.parse_args()



prepare_data(path=args.smiles_dataset, save_to=args.selfies_dataset)
chembl_df = pd.read_csv(args.selfies_dataset)
print("SELFIES representation file is ready.")