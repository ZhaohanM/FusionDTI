import json
import re
import torch
import torch.nn as nn
from torch.nn import functional as F

class DrugTokenizer:
    def __init__(self, vocab_path="data/Tokenizer/vocab.json", special_tokens_path="data/Tokenizer/special_tokens_map.json"):
        self.vocab, self.special_tokens = self.load_vocab_and_special_tokens(vocab_path, special_tokens_path)
        self.cls_token_id = self.vocab[self.special_tokens['cls_token']]
        self.sep_token_id = self.vocab[self.special_tokens['sep_token']]
        self.unk_token_id = self.vocab[self.special_tokens['unk_token']]
        self.pad_token_id = self.vocab[self.special_tokens['pad_token']]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def load_vocab_and_special_tokens(self, vocab_path, special_tokens_path):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)
        with open(special_tokens_path, 'r', encoding='utf-8') as special_tokens_file:
            special_tokens_raw = json.load(special_tokens_file)

        special_tokens = {key: value['content'] for key, value in special_tokens_raw.items()}
        return vocab, special_tokens

    def encode(self, sequence):
        tokens = re.findall(r'\[([^\[\]]+)\]', sequence)
        input_ids = [self.cls_token_id] + [self.vocab.get(token, self.unk_token_id) for token in tokens] + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def batch_encode_plus(self, sequences, max_length, padding, truncation, add_special_tokens, return_tensors):
        input_ids_list = []
        attention_mask_list = []

        for sequence in sequences:
            encoded = self.encode(sequence)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            elif len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [self.vocab[self.special_tokens['pad_token']]] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
        }

    def decode(self, input_ids, skip_special_tokens=False):
        tokens = []
        for id in input_ids:
            if skip_special_tokens and id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            tokens.append(self.id_to_token.get(id, self.special_tokens['unk_token']))
        sequence = ''.join([f'[{token}]' for token in tokens])
        return sequence
