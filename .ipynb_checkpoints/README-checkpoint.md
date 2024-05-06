# TF-DTI
TF-DTI utilises a Token-level Fusion module to effectively learn fine-grained information for Drug-Target Interaction Prediction.
## Framework
![TF-DTI](image/TF-DTI.jpg)

## Installation Guide
Clone this Github repo and set up a new conda environment.
```
# create a new conda environment
$ conda create --name TF-DTI python=3.8
$ conda activate TF-DTI

# install requried python dependencies
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install --upgrade transformers
$ pip install wandb

# clone the source code of TF-DTI
$ git https://github.com/ZhaohanM/TF-DTI.git
$ cd TF-DTI
```
## Datasets
All data used in TF-DTI are from public resource: [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Human](https://github.com/lifanchen-simm/transformerCPI) [3]. The dataset can be downloaded from [here](https://github.com/peizhenbai/DrugBAN/tree/main/datasets).

## Run TF-DTI on Our Data to Reproduce Results

For the experiments with TF-DTI, you can directly run the following command. The dataset could either be  `BindingDB`, `Biosnap`, and `Human`. 
```
$ python main.py --dataset BindingDB

``` 
## How to obtain the structure-aware sequence of protein?

The structure-aware sequence of protein is based on 3D structure file (.cif) using Foldseek from the [AlphafoldDB](https://alphafold.ebi.ac.uk) database.
[SaProt](https://github.com/westlake-repl/SaProt?tab=readme-ov-file) provides a function to convert a protein structure into a structure-aware sequence. The function calls the [foldseek](https://github.com/steineggerlab/foldseek) binary file to encode the structure. You can download the binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing) and place it in the `utils` folder. 

The following three steps are the obtainment process:

The first step, if you do not have Uniprot IDs, you will need to obtain them from the [Uniprot website](https://www.uniprot.org) based on existing amino acid sequences, protein names, etc. Then save them as a comma-delimited text file.

In the second step, the following code is run to get the protein structure file corresponding to the Uniprot ID.
```
$ python get_alphafold.py

```
Finally, you can run the following code to retrieve the structure-aware sequence of the protein.
```
$ python generate_stru_seq.py

```
## How to obtain SELFIES of drug?

You need to install the python packages that convert the drug SMILES strings into SELFIES strings.
```
$ pip install selfies 
$ pip install pandarallel
```
Run the following code to generate SELFIES based on your SMILES.
```
$ python generate_selfies.py

```