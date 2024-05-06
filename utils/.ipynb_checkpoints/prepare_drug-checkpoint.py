import pandas as pd
from pandarallel import pandarallel
import selfies as sf


def to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        return None

def prepare_data(path="./dataset/BindingDB.csv", save_to="./dataset/BindingDB_SEFLIES.csv"):
    drugbank_df = pd.read_csv(path)
    drugbank_df["selfies"] = drugbank_df["SMILES"].copy()

    pandarallel.initialize()
    drugbank_df["selfies"] = drugbank_df["selfies"].parallel_apply(to_selfies)

    # Drop rows where SMILES couldn't be converted to SELFIES
    drugbank_df.dropna(subset=["selfies"], inplace=True)

    drugbank_df.to_csv(save_to, index=False)
    

def create_selfies_file(selfies_df, save_to="./dataset/selfies_subset.txt", subset_size=100000, do_subset=True):
    selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling

    if do_subset:
        selfies_subset = selfies_df.selfies[:subset_size]
    else:
        selfies_subset = selfies_df.selfies
    selfies_subset = selfies_subset.to_frame()
    selfies_subset["selfies"].to_csv(save_to, index=False, header=False)

    
