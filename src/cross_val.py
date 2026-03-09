
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch.multiprocessing
#from bigquery import load_table, save_table

import pytorch_lightning as pl
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import hydra
from omegaconf import DictConfig 
from config.definitions import ROOT_DIR

from data.datamodule import Note_Data_Module
from model import Note_Classifier

from metrics import evaluate
from sklearn import metrics 

from sklearn.model_selection import ParameterGrid
from preprocessing.split import get_data_and_vocab


RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system') # see if it corrects the issue by resuming the training 

@hydra.main(
        config_path = "./config",
        version_base = None, 
        config_name = "config_cv"
)
def cross_val(cfg: DictConfig): 
    
    args = cfg.args
    model_config = cfg.model_config 


    hyperparameters_dict = {
        'n_epochs': [1],
        'weight_decay': [0.001], #[0.01, 0.05, 0.1],
        'lr': [1.5e-6, 2e-5], #[1e-5, 5e-5, 1e-4, 5e-4],
        #"adam_epsilon": [1e-6],
        # "warmup_steps": [50],
        'batch_size': [32], # [8]
        'warmup': [0.2], 
        # "save_steps": [np.inf]
    }
    hyperparameters_list = list(ParameterGrid(hyperparameters_dict))
    print(hyperparameters_list)
    runs_ids = []

    # Add run ids to identify each run
    for i in range(len(hyperparameters_list)):
        hyperparameters_list[i]['run_id'] = i
        runs_ids.append(hyperparameters_list[i]['run_id'])


    # model 
    model_config['num_workers'] = int(os.cpu_count() - 4) # Multi cpu workers

    # Make one directory under results for each run (each combination of hyperparameters)
    for i, run_id in enumerate(runs_ids):
        curr_path = 'results/{}_Run{}'.format(args.cv_name, run_id)
        if not os.path.exists(curr_path):
            print("AAA")
            os.mkdir(curr_path)
        
    
    preprocessed_data_folder = os.path.join(ROOT_DIR, args.preprocessed_data_folder)
    preprocessed_data_folder

    for i, run_id in enumerate(runs_ids):
        
        print(f"##############################################")
        print(f"Starting run {run_id+1}/{len(runs_ids)}...")
        hyperparameters = hyperparameters_list[i]
        
        # Train on each fold
        for k in range(args.n_folds):
            
            print(f"\t---> Fold number {k+1}/{args.n_folds}:")
            save_name = "{}_Run{}/Fold_{}".format(args.cv_name, run_id, k)

            fold_path = os.path.join(preprocessed_data_folder, f"fold_{k+1}")
            train_df, val_df, vocab  = get_data_and_vocab(os.path.join(fold_path, "train"), os.path.join(fold_path, "train") )

            model_config['train_size'] = len(train_df)

            note_data_module = Note_Data_Module(train_df, 
                                        val_df, 
                                        classes=model_config["classes"], 
                                        batch_size=model_config['batch_size'], 
                                        max_token_length=model_config['max_token_length'],
                                        model_name=model_config['model_name'],
                                        return_overflowing_tokens=model_config['return_overflowing_tokens'], 
                                        num_workers = model_config['num_workers']
                                        )
            note_data_module.setup()

            model_config.update(hyperparameters)

            model = Note_Classifier(model_config)

            checkpoint_callback = ModelCheckpoint(
                dirpath=f"{save_name}-checkpoints",
                filename="best-checkpoint",
                save_top_k=1,
                verbose=True,
                monitor="val_loss",
                mode="min"
            )

            logger = TensorBoardLogger("lightning_logs", name=save_name)
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)


            trainer = pl.Trainer(
                logger=logger,
                callbacks=[checkpoint_callback, early_stopping_callback],
                max_epochs=model_config['n_epochs'],
                gpus=0, 
                num_sanity_val_steps=0, 
                auto_lr_find=False, 
                deterministic=False, 
                fast_dev_run=False
            )
            
            trainer.fit(model, note_data_module)

        print(f"##############################################")


if __name__ == "__main__":
    cross_val()