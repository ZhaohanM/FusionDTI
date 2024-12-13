## Datasets

All data used in FusionDTI are from public resources: [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Human](https://github.com/lifanchen-simm/transformerCPI) [3]. The dataset can be downloaded from [here](https://github.com/peizhenbai/DrugBAN/tree/main/datasets).

## References
    [1] Liu, Tiqing, Yuhmei Lin, Xin Wen, Robert N. Jorissen, and Michael K. Gilson (2007). BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities. Nucleic acids research, 35(suppl_1), D198-D201.
    [2] Huang, Kexin, Cao Xiao, Lucas M. Glass, and Jimeng Sun (2021). MolTrans: Molecular Interaction Transformer for drug–target interaction prediction. Bioinformatics, 37(6), 830-836.
    [3] Chen, Lifan, et al (2020). TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinformatics, 36(16), 4406-4414.

## processed_DTI_Token

The joint embedding vectors of the datasets are stored in the `processed_DTI_Token` folder for later-stage MLP training. 

## How to obtain the structure-aware sequence of protein?

The structure-aware sequence of the protein is based on the 3D structure file (.cif) using Foldseek from the [AlphafoldDB](https://alphafold.ebi.ac.uk) database.
[SaProt](https://github.com/westlake-repl/SaProt?tab=readme-ov-file) provides a function to convert a protein structure into a structure-aware sequence. The function calls the [foldseek](https://github.com/steineggerlab/foldseek) binary file to encode the structure. You can download the binary file from [here](https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view?usp=sharing) and place it in the `utils` folder. 

The following three steps are the obtainment process:

The first step, if you do not have Uniprot IDs, you will need to obtain them from the [Uniprot website](https://www.uniprot.org) based on existing amino acid sequences, protein names, etc. Then save them as a comma-delimited text file.

The following code is run in the second step to get the protein structure file corresponding to the Uniprot ID.
```
$ python get_alphafold.py
```
Finally, you can run the following code to retrieve the structure-aware sequence of the protein.
```
$ python generate_stru_seq.py
```
## How to obtain SELFIES of drug?

You must install the Python packages that convert the drug SMILES strings into SELFIES strings.
```
$ pip install selfies 
$ pip install pandarallel
```
Run the following code to generate SELFIES based on your SMILES.
```
$ python generate_selfies.py
```
