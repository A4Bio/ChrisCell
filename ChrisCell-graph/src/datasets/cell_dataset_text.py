import torch.utils.data as data
import numpy as np
from transformers import AutoTokenizer
from src.modules.pifold_module_sae import GeoFeaturizer
import scanpy as sc
import torch
import os
from tqdm import tqdm
from time import time
import random
from transformers import AutoTokenizer

def list_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

class ADataset(data.Dataset):
    def __init__(self, data_path='./',  split='train', data=None, gene_dim=None, device=0, use_property=False):
        self.__dict__.update(locals())
        self.device = 0
        self.adata = []
        self.split = split
        self.device_num = torch.cuda.device_count()
        self.use_property = use_property
        if split == 'train':
            self.lengths = []
            if data is None:
                if not os.path.isfile(data_path):
                    files = list_all_files(data_path)
                else:
                    files = [data_path]
                    
                num = len(files) // self.device_num
                files = files[num*device: (device+1)*num]
                for file in tqdm(files):
                    adata = sc.read(file)
                    self.lengths.append(adata.shape[0])
                    self.adata.append(adata)
            else:
                self.adata = data
            self.all_len = sum(self.lengths)
            self.len = self.all_len
            if torch.cuda.device_count() > 1:
                self.len = 20000000
        else:
            self.adata = [data[0]]
            self.lengths = [10]
            self.all_len = sum(self.lengths)
            self.len = self.all_len
        self.gene_num = self.adata[0].shape[-1]
        self.cell_types = []
        for adata in self.adata:
            self.cell_types+=adata.obs['celltype'].unique().tolist() if 'celltype' in adata.obs else adata.obs['cell_type'].unique().tolist()
        self.cell_types = list(set(self.cell_types))
        if use_property:
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
        print('data preprocessed')
        
    
    def __len__(self):
        return self.len
    
    def _get_data(self, index):
        i = 0
        while index >= self.lengths[i]:
            index -= self.lengths[i]
            i+=1
        data = self.adata[i][index]
        X = data.X.toarray()

        return X, None, None

    def get_data(self, index):
        i = 0
        while index >= self.lengths[i]:
            index -= self.lengths[i]
            i+=1
        data = self.adata[i][index]
        extract_feature_names = ['tissue', 'cell_type', 'disease', 'development_stage','sex']
        feature_sequence = []
        feature_indexes = []
        for i, feature_name in enumerate(extract_feature_names):
            feature = self.tokenizer.encode(data.obs[feature_name][0], add_special_tokens=False)
            feature_sequence += feature
            feature_indexes += len(feature)*[i]
        if len(feature_sequence) < 30:
            feature_sequence += [self.tokenizer.pad_token_id] * (30 - len(feature_sequence))
            feature_indexes += [-1] * (30 - len(feature_indexes))

        return data.X.toarray(), torch.LongTensor(feature_sequence[:30]), torch.LongTensor(feature_indexes[:30])
    
    
    def __getitem__(self, index):
        if self.split == 'train' and self.device_num > 1:
            index = random.randint(0, self.all_len-1)
        if self.use_property:
            X, feature_sequence, feature_indexes = self.get_data(index)
            feature_sequence = feature_sequence.unsqueeze(0)
            feature_indexes = feature_indexes.unsqueeze(0)
        else:
            X, feature_sequence, feature_indexes = self._get_data(index)

        data = {'X': torch.from_numpy(X),
                'properties': feature_sequence,
                'property_index': feature_indexes}
        return data
