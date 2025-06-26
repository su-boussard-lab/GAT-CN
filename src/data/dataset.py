from torch.utils.data import Dataset
import torch


class NoteDataset(Dataset):
    def __init__(self, data, tokenizer, classes, max_token_length: int = 512, return_overflowing_tokens: bool = False): 
        self.data = data 
        self.tokenizer = tokenizer
        self.classes = classes
        self.max_token_length = max_token_length
        self.return_overflowing_tokens = return_overflowing_tokens
        
    def __len__(self):
        return len(self.data)
    
    def tokenize_note(self, note):
        tokens = self.tokenizer.encode_plus(note, 
                                          add_special_tokens=True, 
                                          return_tensors='pt',
                                          truncation=True, 
                                          padding='max_length', 
                                          max_length=self.max_token_length, 
                                          return_attention_mask=True,
                                          return_overflowing_tokens=self.return_overflowing_tokens)
        return tokens 
    
        
    

class NoteBertDataset(NoteDataset):
    
    def __init__(self, data, tokenizer, classes, max_token_length: int = 512, return_overflowing_tokens: bool = False): 
        super().__init__(
            data, 
            tokenizer, 
            classes, 
            max_token_length,
            return_overflowing_tokens
        ) 
        self.data = data 
        self.tokenizer = tokenizer
        self.classes = classes
        self.max_token_length = max_token_length
        self.return_overflowing_tokens = return_overflowing_tokens
    
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        classes = torch.FloatTensor(item[self.classes])
        tokens = self.tokenize_note(item.NOTE)
        
        if self.return_overflowing_tokens: 
            num_chunks = tokens.input_ids.shape[0]
            classes = torch.FloatTensor([item[self.classes]]*num_chunks) 
            return {'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask, 'labels': classes}
        else: 
            return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': classes}
    
    

class NoteBertGnnDataset(NoteDataset):
    
    def __init__(self, data, adj_dict, tokenizer, classes, max_token_length: int = 512, return_overflowing_tokens: bool = False): 
        super().__init__(
            data, 
            tokenizer, 
            classes, 
            max_token_length,
            return_overflowing_tokens
        ) 
        self.adj_dict = adj_dict
        

    def __getitem__(self, index):
        item = self.data.iloc[index]
        tokens = self.tokenize_note(item.NOTE)

        if self.return_overflowing_tokens: 
            num_chunks = tokens.input_ids.shape[0]
            label = torch.FloatTensor([item[self.classes].values.tolist()]*num_chunks) 
            edges = torch.FloatTensor([self.adj_dict[index].tolist()]*num_chunks)
            return {'input_ids': tokens.input_ids.type(torch.long), 'edges': edges, 'labels': label}
        else: 
            return {'input_ids': tokens.input_ids.flatten().type(torch.long) , 'edges': torch.tensor(self.adj_dict[index].tolist(), dtype=torch.double), 'labels':  torch.tensor(item[self.classes], dtype=torch.long)}
    