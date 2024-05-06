# import os
# import time
# import json
# import numpy as np
# import sys
# sys.path.append(".")


# # Get structural seqs from pdb file
# def get_struc_seq(foldseek,
#                   path,
#                   chains: list = None,
#                   process_id: int = 0,
#                   plddt_path: str = None,
#                   plddt_threshold: float = 70.) -> dict:
#     """
    
#     Args:
#         foldseek: Binary executable file of foldseek
#         path: Path to pdb file
#         chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
#         process_id: Process ID for temporary files. This is used for parallel processing.
#         plddt_path: Path to plddt file. If None, plddt will not be used.
#         plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

#     Returns:
#         seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
#         (seq, struc_seq, combined_seq).
#     """
#     assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
#     assert os.path.exists(path), f"Pdb file not found: {path}"
#     assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
#     tmp_save_path = f"get_struc_seq_{process_id}.tsv"
#     cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
#     os.system(cmd)

#     seq_dict = {}
#     name = os.path.basename(path)
#     with open(tmp_save_path, "r") as r:
#         for i, line in enumerate(r):
#             desc, seq, struc_seq = line.split("\t")[:3]
            
#             # Mask low plddt
#             if plddt_path is not None:
#                 with open(plddt_path, "r") as r:
#                     plddts = np.array(json.load(r)["confidenceScore"])
                    
#                     # Mask regions with plddt < threshold
#                     indices = np.where(plddts < plddt_threshold)[0]
#                     np_seq = np.array(list(struc_seq))
#                     np_seq[indices] = "#"
#                     struc_seq = "".join(np_seq)
            
#             name_chain = desc.split(" ")[0]
#             chain = name_chain.replace(name, "").split("_")[-1]

#             if chains is None or chain in chains:
#                 if chain not in seq_dict:
#                     combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
#                     seq_dict[chain] = (seq, struc_seq, combined_seq)
        
#     os.remove(tmp_save_path)
#     os.remove(tmp_save_path + ".dbtype")
#     return seq_dict


# if __name__ == '__main__':
#     foldseek = "/sujin/bin/foldseek"
#     # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
#     test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
#     plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
#     res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
#     print(res["A"][1].lower())

import os
import json
import numpy as np
import subprocess

def get_struc_seq(foldseek, 
                  path, 
                  chains: list = None, 
                  process_id: int = 0, 
                  plddt_path: str = None, 
                  plddt_threshold: float = 70.) -> dict:
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
    tmp_save_base = f"get_struc_seq_{process_id}"
    cmd = [foldseek, "structureto3didescriptor", "-v", "0", "--threads", "1", "--chain-name-mode", "1", path, tmp_save_base]

    # 使用 subprocess 运行命令
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"Error running foldseek: {process.stderr.decode()}")
        return {}

    # 读取和处理所有相关的输出文件
    seq_dict = {}
    for file_name in os.listdir("."):
        if file_name.startswith(tmp_save_base) and not file_name.endswith(".dbtype"):
            with open(file_name, "r") as file:
                # 处理每个文件的内容
                # 这里需要根据实际文件格式来调整
                for line in file:
                    desc, seq, struc_seq = line.split("\t")[:3]
                    if plddt_path is not None:
                        with open(plddt_path, "r") as r:
                            plddts = np.array(json.load(r)["confidenceScore"])
                            indices = np.where(plddts < plddt_threshold)[0]
                            np_seq = np.array(list(struc_seq))
                            np_seq[indices] = "#"
                            struc_seq = "".join(np_seq)

                    name_chain = desc.split(" ")[0]
                    chain = name_chain.split("_")[-1]

                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                            seq_dict[chain] = (seq, struc_seq, combined_seq)

            # 删除处理过的文件
            os.remove(file_name)

    return seq_dict

# 示例调用（请根据需要进行修改）
if __name__ == '__main__':
    foldseek = "/bin/foldseek"  # 更新为 foldseek 的实际路径
    # test_path = "/path/to/pdb_file.pdb"  # 更新为 PDB 文件的实际路径
    plddt_path = "/path/to/plddt_file.json"  # 更新为 PLDDT 文件的实际路径（如果有的话）
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    # print(res)
