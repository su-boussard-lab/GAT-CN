from transformers import AutoModel,  AutoTokenizer, LongformerModel, AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import pandas as pd
import torch 
import torchmetrics
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_scatter import scatter_mean, scatter_max
from torch.optim.lr_scheduler import ExponentialLR
from scipy.sparse import coo_matrix
import dgl
import dgl.nn.pytorch as dglnn
from torch.utils.data import DataLoader
from model.Graph_Conv import *



class NoteClassifier(pl.LightningModule):
    
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.n_labels = self.config.n_labels
        self.threshold = self.config.threshold
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
        
        self.lr = self.config.optimizer.lr
        self.batch_size = self.config.batch_size
        self.dropout_p = self.config.regularization.dropout_p 
        
        self.cls_pooling = self.config.bert.cls_pooling
        self.return_sequence = self.config.bert.return_sequence
        
        if self.config.bert.longformer:
            self.pretrained_model = LongformerModel.from_pretrained(self.config.bert.model_name, return_dict=True)
        else: 
            self.pretrained_model = AutoModel.from_pretrained(self.config.bert.model_name, return_dict=True)
        if self.config.bert.longformer:
            self.tokenizer = LongformerModel.from_pretrained(self.config.bert.model_name, return_dict=True)
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert.model_name) 
        
        if self.config.loss.with_imbalance: 
            self.loss_fct = nn.BCEWithLogitsLoss(weight=torch.tensor(self.config.loss.class_weight), reduction="mean", pos_weight=torch.tensor(self.config.loss.pos_weight))

        # Training metrics 
        self.accuracy_micro_train = torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="micro")
        self.accuracy_macro_train= torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="macro")
        self.accuracy_weighted_train = torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="weighted")

        # Val metrics 
        self.accuracy_micro_val = torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="micro")
        self.accuracy_macro_val= torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="macro")
        self.accuracy_weighted_val = torchmetrics.Accuracy(task="multilabel", threshold=self.threshold, num_labels=self.n_labels, average="weighted")
        
    
    def postprocess(
        self,
        embeddings: torch.Tensor,
        step_chunks: torch.Tensor,
        cls_pooling: bool = True,
        return_sequence: bool = False,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Postprocesses the output embeddings, such that they correspond to their input dimension
        Average the chunks either on the CLS token or pool them generally
            Input Dim: [sum chunks, emb_dim]
            Output Dim: [batchsize, emb_dim]
        Args:
            embeddings (torch.Tensor): tensor of the embeddings
            attention_masks (torch.Tensor): tensor of the attention masks
            step_chunks (torch.Tensor): tensor containing the chunk indices per note
            cls_pool (bool, True): if averaging the chunks only on the CLS token or over the whole note
            return_sequence (bool, False): returns the full sequence rather than only the CLS or averaged token
        returns:
            new_dicts (dict): new shape: [sum chunks in the whole batch, max_length]
        """
        if return_sequence:
            return scatter_mean(embeddings, step_chunks, dim=0)
        
        # Pool only the CLS token at the beggining of each sentence
        if cls_pooling:
            embeddings = embeddings[:, 0, :]

        # Pool all the sequences together and take their average weighted by the mask
        elif attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            )
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)   
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
 

        embeddings_pooled = scatter_mean(embeddings, step_chunks, dim=0)
        return embeddings_pooled

     
    def training_step(self, batch, batch_index):
        loss, logits, labels = self(batch)  
        self.log("step", self.global_step, logger=True, on_step=True)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log step metric
        self.accuracy_micro_train(logits, labels)
        self.accuracy_macro_train(logits, labels)
        self.accuracy_weighted_train(logits, labels)
        self.log('train_acc_micro', self.accuracy_micro_train, on_step=True, on_epoch=True)
        self.log('train_acc_macro', self.accuracy_macro_train, on_step=True, on_epoch=True)
        self.log('train_acc_weighted', self.accuracy_weighted_train, on_step=True, on_epoch=True)

        return {"loss": loss, "predictions": logits, "labels": labels}
    
    
    def validation_step(self, batch, batch_index):
        loss, logits, labels = self(batch) 
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # log step metric
        self.accuracy_micro_val(logits, labels)
        self.accuracy_macro_val(logits, labels)
        self.accuracy_weighted_val(logits, labels)
        self.log('val_acc_micro', self.accuracy_micro_val, on_step=True, on_epoch=True)
        self.log('val_acc_macro', self.accuracy_macro_val, on_step=True, on_epoch=True)
        self.log('val_acc_weighted', self.accuracy_weighted_val, on_step=True, on_epoch=True)

        return {"val_loss": loss, "predictions": logits, "labels": labels}


    def predict_step(self, batch, batch_index):
        _, logits, _ = self(batch)
        return logits
    
    def configure_optimizers(self):
        params_1x = [param for name, param in self.named_parameters() if "pretrained_model." not in name]
        if self.config.optimizer.split_lr: 
            optimizer  = AdamW([{'params': params_1x},
                                {'params': self.pretrained_model.parameters(),
                                'lr': self.lr * 10}],
                                lr=self.lr, weight_decay=self.config.optimizer.weight_decay)   
        else: 
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.config.optimizer.weight_decay)
        
        steps_per_epoch = self.config.train_size / self.batch_size
        total_training_steps = steps_per_epoch * self.config.n_epochs
        warmup_steps = math.floor(total_training_steps * self.config.optimizer.warmup)
        if self.config.optimizer.scheduler == "cosine": 
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                )
            )
        elif self.config.optimizer.scheduler == "linear": 
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                )
            )   
        elif self.config.optimizer.scheduler == "reduce_on_plateau": 
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
            return dict(
            optimizer=optimizer,
            lr_scheduler= {'scheduler': scheduler, 'monitor': 'val_loss'}
            )
        elif self.config.optimizer.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.5) #for gnn
            return [optimizer], [scheduler]
  
  
    def preprocess_dataframe(self, data, dic_path, resize_token_embeddings):
        # adj
        dic = pd.read_pickle(os.path.join(os.path.abspath(os.path.join(dic_path, os.pardir)), data.dic_feature_path)) # dictionary
        self.dic_feature = torch.tensor(dic['feature'].tolist()) 
        self.dic_dic = torch.tensor(np.load(os.path.join(dic_path, data.dic_dic)) ,dtype=torch.double) # word-word pmi
        
        #add token
        if resize_token_embeddings: 
            words = dic['lexicon'].tolist()
            print("vocab size (before) : ", len(self.tokenizer))
            for w in words:
                self.tokenizer.add_tokens(w, special_tokens=True)
            print("vocab size (after) : ", len(self.tokenizer))
            self.pretrained_model.resize_token_embeddings(len(self.tokenizer))

    

class NoteBertClassifier(NoteClassifier,):
    def __init__(self, config: dict, classes: list):
        super().__init__(config)
        
        #self.pretrained_model = AutoModel.from_pretrained(self.config.bert.model_name, return_dict=True)
        self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.n_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        self.classes = classes
        
        
    def forward(self, batch):
        
        if type(batch) is not dict:
            input_dict, step_chunks = batch
        else: 
            input_dict = batch
            step_chunks = torch.Tensor(np.arange(len(batch["input_ids"]))).to(dtype=torch.int64) 
        input_ids, attention_mask, labels = (v for k, v in input_dict.items())
        
        # bert layer
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = output.last_hidden_state
        pooled_output = self.postprocess(
                pooled_output,
                step_chunks,
                cls_pooling=self.cls_pooling,
                return_sequence=self.return_sequence,
                attention_mask=attention_mask,
            ) 
        
        if self.return_sequence: 
            pooled_output = torch.mean(output.last_hidden_state, 1) 
            
        # final logits
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # calculate loss
        loss = 0
        if labels is not None: 
            labels = scatter_max(labels, step_chunks, dim=0)[0]
            loss = self.loss_fct(logits.view(-1, self.n_labels), labels.view(-1, self.n_labels))

        return loss, logits, labels
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {'avg_val_loss': avg_val_loss} 
    
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []

        for output in outputs: 
            for out_labels in output['labels']: #.detach().cpu()
                labels.append(out_labels)

            for out_predictions in output['predictions']: #.detach().cpu()
                predictions.append(out_predictions)

        y_true = [np.array(label).tolist() for label in labels]
        flattened_predictions = [torch.sigmoid(prediction).detach().numpy() for prediction in predictions]
        y_pred = [np.where(np.array(flattened_prediction) > self.threshold , 1., 0.).tolist() for flattened_prediction in flattened_predictions]

        report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    target_names=self.classes
                )

        labels = torch.stack(labels)
        predictions = torch.stack(predictions)
        for i, name in enumerate(self.classes):
            roc_score = auroc(predictions[:, i], labels[:, i].long(),  task ='binary')
            self.log(f"{name}_roc_auc/Train", roc_score, logger=True, on_epoch=True)
            #self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", roc_score, self.current_epoch)
            self.log(f"{name}_precision/Train", report[name]['precision'], logger=True, on_epoch=True)
            self.log(f"{name}_recall/Train", report[name]['recall'], logger=True, on_epoch=True)
            self.log(f"{name}_f1_score/Train", report[name]['f1-score'], logger=True, on_epoch=True)
        
        self.log(f"learning_rate", self.lr, logger=True, on_epoch=True)
            
            
            
        
        
        
        
class NoteBertGnnClassifier(NoteClassifier):
    def __init__(self, model_config):
        super().__init__(model_config)
        
        #tuning
        self.agg_type = self.config.gnn.agg
        self.hidden = self.config.gnn.hidden
        self.s_drop = self.config.gnn.s_drop 
        self.kernel_out = self.config.gnn.kernel_out
        self.dic_hidden = self.config.gnn.dic_hidden
        
        # graphsage      
        if self.config.gnn.post_word_edge:
            self.sampler = dgl.dataloading.MultiLayerNeighborSampler([
                        {('dic', 'co-occur', 'dic'): self.config.gnn.dic_edge_1,
                         ('dic', 'in', 'post'):  self.config.gnn.dic_post_edge_1,
                         ('post', 'contains', 'dic'):  self.config.gnn.post_dic_edge_1},
                        {('dic', 'co-occur', 'dic'): self.config.gnn.dic_edge_2,
                         ('dic', 'in', 'post'): self.config.gnn.dic_post_edge_2,
                         ('post', 'contains', 'dic'): self.config.gnn.post_dic_edge_2}
                    ])
        else: 
            self.sampler = dgl.dataloading.MultiLayerNeighborSampler([
                            {('dic', 'co-occur', 'dic'):self.config.gnn.dic_edge_1,
                            ('dic', 'in', 'post'): self.config.gnn.dic_post_edge_1},
                            {('dic', 'co-occur', 'dic'): self.config.gnn.dic_edge_2,
                            ('dic', 'in', 'post'): self.config.gnn.dic_post_edge_2}
                        ])
        
        if self.config.gnn.post_word_edge:
            self.conv1 = dglnn.HeteroGraphConv({
                'in' : SAGEConv((201,self.pretrained_model.config.hidden_size), self.hidden, aggregator_type= self.agg_type, feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                'co-occur' : SAGEConv((201,201), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                'contains' : SAGEConv((self.pretrained_model.config.hidden_size, 201), self.dic_hidden, aggregator_type= self.agg_type, feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes)},
                aggregate='sum').double()
            
            self.conv2 = dglnn.HeteroGraphConv({
                'in' : SAGEConv((self.dic_hidden, self.hidden), int(self.hidden/2), aggregator_type= self.agg_type, feat_drop=self.s_drop, kernel_out = self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                'co-occur' : SAGEConv((self.dic_hidden, self.dic_hidden), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                'contains' : SAGEConv((self.hidden, self.dic_hidden), self.dic_hidden, aggregator_type= self.agg_type, feat_drop=self.s_drop, kernel_out = self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes)},
                aggregate='sum').double()
        else: 
            self.conv1 = dglnn.HeteroGraphConv({
                    'in' : SAGEConv((201,self.pretrained_model.config.hidden_size), self.hidden, aggregator_type= self.agg_type, feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                    'co-occur' : SAGEConv((201,201), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes)},
                    aggregate='sum').double()
            
            self.conv2 = dglnn.HeteroGraphConv({
                    'in' : SAGEConv((self.dic_hidden, self.hidden), int(self.hidden/2), aggregator_type= self.agg_type, feat_drop=self.s_drop, kernel_out = self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes),
                    'co-occur' : SAGEConv((self.dic_hidden, self.dic_hidden), self.dic_hidden, aggregator_type=self.agg_type,feat_drop=self.s_drop,kernel_out=self.kernel_out, kernel_sizes=self.config.gnn.kernel_sizes)},
                    aggregate='sum').double()
        
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = torch.nn.Linear(int(self.hidden/2), self.config.n_labels)
        
    
    def forward(self, batch, **kwargs):
        if type(batch) is not dict:
            input_dict, step_chunks = batch
        else: 
            input_dict = batch
            step_chunks = torch.Tensor(np.arange(len(batch["input_ids"]))).to(dtype=torch.int64) 
            
        text_data, edges, labels = (v for k, v in input_dict.items())
        
        labels = scatter_max(labels, step_chunks, dim=0)[0]
        edges = scatter_max(edges, step_chunks, dim=0)[0]
     
        # post embedding
        outputs_data = self.pretrained_model(input_ids=text_data, **kwargs) 

        pooled_output = outputs_data.last_hidden_state

        p_feat = self.postprocess(
                pooled_output,
                step_chunks,
                cls_pooling=self.cls_pooling,
            ) 
            
        
        #graph
        edge_index = torch.nonzero(edges, as_tuple=False).T
        dic_index = torch.nonzero(self.dic_dic, as_tuple=False).T 
        
        if self.config.gnn.post_word_edge:
            g = dgl.heterograph(data_dict = {('dic', 'in', 'post')  : (edge_index[1], edge_index[0]),
                                            ('post', 'contains', 'dic') :(edge_index[0], edge_index[1]),
                                            ('dic', 'co-occur', 'dic')  : (dic_index[0], dic_index[1])},
                            num_nodes_dict = {'dic':len(self.dic_feature), 'post':len(p_feat)}) 
        else:
            g = dgl.heterograph(data_dict = {('dic', 'in', 'post')  : (edge_index[1], edge_index[0]),
                                            ('dic', 'co-occur', 'dic')  : (dic_index[0], dic_index[1])},
                            num_nodes_dict = {'dic':len(self.dic_feature), 'post':len(p_feat)}) 
            
        d_feat = torch.tensor(self.dic_feature.numpy().astype(np.float32),dtype = torch.double) 
        
        g.ndata['features'] = {'post' : p_feat.double(), 'dic' : d_feat}
        if self.config.gnn.post_word_edge:
            g.edata['features'] = {'co-occur' : torch.tensor(coo_matrix(self.dic_dic).data), 
                               'in':torch.tensor(coo_matrix(edges).data), 
                               'contains':torch.tensor(coo_matrix(edges).data)}
        else:
            g.edata['features'] = {'co-occur' : torch.tensor(coo_matrix(self.dic_dic).data), 
                                   'in':torch.tensor(coo_matrix(edges).data)} 
        
        train_nid = {'post': torch.tensor(range(len(p_feat))), 
                    'dic': torch.tensor(range(len(self.dic_feature)))} 
        #, 
        
        collator = dgl.dataloading.NodeCollator(g, train_nid, self.sampler)
        dataloader = DataLoader(
            collator.dataset, collate_fn=collator.collate,
            batch_size=int(g.number_of_nodes()/5), shuffle=False, drop_last=False) 
        
        post_output = None
        for i,(input_nodes, output_nodes, blocks) in enumerate(dataloader):
            input_features = blocks[0].srcdata['features']     
            output = self.conv1(blocks[0], input_features,blocks[0].edata['features'])   
            output = self.conv2(blocks[1], output,blocks[1].edata['features'])

            if 'post' in output:
                if post_output is None:
                    post_output = output['post']
                else:
                    post_output = torch.cat([post_output,output['post']],dim=0)
                    
        
        del collator
        del dataloader

        logits = self.classifier(self.dropout(post_output.float()))
        
        # calculate loss
        loss = 0
        if labels is not None: 
            loss = self.loss_fct(logits.view(-1, self.config.n_labels), labels.view(-1, self.config.n_labels).float())

        return loss, logits, labels
        


   