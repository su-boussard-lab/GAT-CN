
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import NoteBertDataset, NoteBertGnnDataset
from transformers import AutoTokenizer, LongformerTokenizer
import torch 
import numpy as np 


class NoteDataModule(pl.LightningDataModule): 
    def __init__(self, train_data, val_data, classes, batch_size = 4, max_token_length = 512, model_name='emilyalsentzer/Bio_ClinicalBERT', return_overflowing_tokens=False, num_workers=4, longformer=False): 
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.classes = classes
        self.batch_size = batch_size
        self.return_overflowing_tokens = return_overflowing_tokens
        self.num_workers = num_workers
        self.max_token_length = max_token_length
        if longformer:
            self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def collate_fn(self, list_input_dicts: dict):
        new_dicts = {}
        input_keys = list_input_dicts[0].keys()
        for key in input_keys:
            stacked_inputs = []
            step_chunks = []
            for i, input_dict in enumerate(list_input_dicts): 
                stacked_inputs.append(input_dict[key])
                step_chunks.extend([i] * len(input_dict[key]))
            # Dim: [sum n_chunks, max_length]
            new_dicts[key] = torch.cat(stacked_inputs, dim=0)
        return new_dicts, torch.Tensor(np.array(step_chunks)).to(dtype=torch.int64)
    
    
    def train_dataloader(self):
        if self.return_overflowing_tokens: 
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)
        else: 
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        if self.return_overflowing_tokens: 
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)
        else: 
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        if self.return_overflowing_tokens: 
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)
        else: 
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
class NoteBertDataModule(NoteDataModule):
    
    def __init__(self, train_data, val_data, classes, batch_size = 4, max_token_length = 512, model_name='emilyalsentzer/Bio_ClinicalBERT', return_overflowing_tokens=False, num_workers=4, longformer=False):
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            classes=classes, 
            batch_size=batch_size,
            max_token_length=max_token_length,
            model_name=model_name,
            return_overflowing_tokens=return_overflowing_tokens,
            num_workers=num_workers,
            longformer=longformer,
        ) 
        
        
    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = NoteBertDataset(self.train_data, tokenizer=self.tokenizer, classes=self.classes,  max_token_length = self.max_token_length, return_overflowing_tokens=self.return_overflowing_tokens)
            self.val_dataset = NoteBertDataset(self.val_data, tokenizer=self.tokenizer, classes=self.classes, max_token_length = self.max_token_length,return_overflowing_tokens=self.return_overflowing_tokens)
        if stage == "predict":
            self.val_dataset = NoteBertDataset(self.val_data, tokenizer=self.tokenizer, classes=self.classes, max_token_length = self.max_token_length, return_overflowing_tokens=self.return_overflowing_tokens)




class NoteBertGnnDataModule(NoteDataModule):
    def __init__(self, train_data, val_data, train_adj_dict, val_adj_dict, classes, batch_size = 4, max_token_length = 512, model_name='emilyalsentzer/Bio_ClinicalBERT', return_overflowing_tokens=False, num_workers=4, longformer=False):
        super().__init__(
            train_data=train_data,
            val_data=val_data,
            classes=classes, 
            batch_size=batch_size,
            max_token_length=max_token_length,
            model_name=model_name,
            return_overflowing_tokens=return_overflowing_tokens,
            num_workers=num_workers,
            longformer=longformer,
        ) 
        self.train_adj_dict = train_adj_dict
        self.val_adj_dict = val_adj_dict
        
    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = NoteBertGnnDataset(
                                   #torch.tensor(self.train_data['token'].values.tolist(), dtype=torch.long),
                                   self.train_data,
                                   self.train_adj_dict,
                                   tokenizer=self.tokenizer, 
                                   classes=self.classes,  
                                   max_token_length = self.max_token_length, 
                                   return_overflowing_tokens=self.return_overflowing_tokens
                                )
            self.val_dataset = NoteBertGnnDataset(
                                   self.val_data,
                                   self.val_adj_dict,
                                   tokenizer=self.tokenizer, 
                                   classes=self.classes,  
                                   max_token_length = self.max_token_length, 
                                   return_overflowing_tokens=self.return_overflowing_tokens
                                )
        if stage == "predict":
            self.val_dataset = NoteBertGnnDataset(
                                    self.val_data,
                                    self.val_adj_dict,
                                    tokenizer=self.tokenizer, 
                                    classes=self.classes,  
                                    max_token_length = self.max_token_length, 
                                    return_overflowing_tokens=self.return_overflowing_tokens
                                )
            
    
