
import numpy as np
import pandas as pd
import os
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils.utils import tmpdir_manager, update_nested_dict
from pytorch_lightning.loggers import NeptuneLogger
import torch
import hydra 
from omegaconf import DictConfig, OmegaConf 
from data.datamodule import NoteBertDataModule, NoteBertGnnDataModule
from model.model import NoteBertClassifier,  NoteBertGnnClassifier
from utils.metrics import evaluate_model
from preprocessing.split import get_data_and_vocab
from config.definitions import ROOT_DIR, DIAGNOSES, CLASSES
from utils.callback import EpochTimeLogger


import sys
sys.path.insert(0, '../preprocessing')

# Set the environment variable TOKENIZERS_PARALLELISM to false
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system') # see if it corrects the issue by resuming the training 


# method to convert list of comments into predictions for each comment
def classify_notes(trainer, model, dm):
   predictions = trainer.predict(model, datamodule=dm)
   flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
   return flattened_predictions 

@hydra.main(
    config_path="./config",
    version_base=None,
    config_name="config_train",
)
def train_model(cfg: DictConfig) -> None:
    args = cfg.args

    # Logger
    run_id = args.logger["run_id"] if args.model.fit.resume_training else None
    logger = NeptuneLogger( 
        project=args.logger.project_name,  
        with_id=run_id,
        api_token=args.logger.api_token,
        description=args.logger.description,
        prefix=args.logger.prefix,
    )  
    
    # Config 
    model_config = cfg.model_config 
    if args.model.fit.resume_training:
        model_config_updated = OmegaConf.to_container(cfg.model_config, resolve=True)
        #update_config = logger.experiment["hyperparams/model_config"].fetch()
        update_nested_dict(model_config_updated, logger.experiment["hyperparams/model_config"].fetch())
        model_config = DictConfig(model_config_updated)
    else:
        logger.log_hyperparams(cfg)
    
    # Data 
    preprocessed_data_folder = os.path.join(ROOT_DIR, model_config.data.preprocessed_data_folder)
    graph_data_folder = os.path.join(ROOT_DIR, model_config.data.graph_data_folder)
    if args.data.testing: 
        train_df, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.full_train_path), os.path.join(preprocessed_data_folder, model_config.data.test_path))
        train_path = os.path.join(graph_data_folder, model_config.data.full_train_path)
        train_adj_dict = np.load(os.path.join(train_path, "doc_word_matrix.npy")) 
        val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.test_path}/doc_word_matrix.npy")) 
    else: 
        train_df, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.train_path), os.path.join(preprocessed_data_folder, model_config.data.val_path))
        train_path = os.path.join(graph_data_folder, model_config.data.train_path)
        train_adj_dict = np.load(os.path.join(train_path, "doc_word_matrix.npy")) 
        val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.val_path}/doc_word_matrix.npy")) 
    model_config.model.train_size = len(train_df) 
      
    # HACK Small - for test
    if args.data.hack: 
        train_df = train_df
        val_df = val_df
        
    if "gnn" in model_config.model.keys(): 
        note_data_module = NoteBertGnnDataModule(train_df, 
                                        val_df, 
                                        train_adj_dict, 
                                        val_adj_dict,
                                        classes=CLASSES, 
                                        batch_size=model_config.model.batch_size, 
                                        max_token_length=model_config.data.max_token_length,
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens,
                                        num_workers=model_config.data.num_workers,
                                        longformer=model_config.model.bert.longformer,
                                        )
        model = NoteBertGnnClassifier(model_config=model_config.model)
        model.preprocess_dataframe(data=model_config.data, dic_path=train_path, resize_token_embeddings=model_config.model.bert.resize_token_embeddings) 
    else: 
        note_data_module = NoteBertDataModule(train_df, 
                                        val_df, 
                                        classes=CLASSES, 
                                        batch_size=model_config.model.batch_size, 
                                        max_token_length=model_config.data.max_token_length,
                                        model_name=model_config.model.bert.model_name,
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens, 
                                        num_workers=model_config.data.num_workers,
                                        longformer=model_config.model.bert.longformer,
                                        )
        model = NoteBertClassifier(model_config.model, classes=CLASSES)
        if model_config.model.bert.resize_token_embeddings:
            model.preprocess_dataframe(data=model_config.data, dic_path=train_path, resize_token_embeddings=model_config.model.bert.resize_token_embeddings) 
            # should we add other words than the one in dic for GNN -> genre give our full vocabulary to incude abbreviations
                           

    print(f'Size of the training and test data: {train_df.shape}, {val_df.shape}')
    
    ## Train model
    # Datamodule
    note_data_module.setup()

        
    # Define callbacks
    callbacks = []
    if model_config.train.callback.checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_config.train.callback.checkpointing.dirpath,
            filename=model_config.train.callback.checkpointing.filename,
            save_top_k=model_config.train.callback.checkpointing.save_top_k,
            save_last=model_config.train.callback.checkpointing.save_last,
            verbose=model_config.train.callback.checkpointing.verbose,
            monitor=model_config.train.callback.checkpointing.monitor,
            mode=model_config.train.callback.checkpointing.mode, 
        )
        callbacks.append(checkpoint_callback)
    if model_config.train.callback.early_stopping_callback:
        early_stopping_callback = EarlyStopping(monitor=model_config.train.callback.early_stopping.monitor,  patience=model_config.train.callback.early_stopping.patience) 
        callbacks.append(early_stopping_callback)
        
    callbacks.append(EpochTimeLogger())
   
    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=model_config.train.logger.log_every_n_steps, 
        max_epochs=model_config.model.n_epochs,
        fast_dev_run=model_config.train.test_mode, 
        profiler=model_config.train.profiler, 
        benchmark=model_config.train.benchmark, 
        callbacks=callbacks if len(callbacks) else None,
        gpus=model_config.train.gpus, 
        precision=16 if model_config.train.fp16 else 32,
        deterministic=model_config.train.deterministic,  # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        num_sanity_val_steps=model_config.train.validate.num_sanity_val_steps, 
        check_val_every_n_epoch=model_config.train.validate.check_val_every_n_epoch,
    )

    # Fit model
    if args.model.fit.resume_training: 
        with tmpdir_manager(base_dir="/tmp") as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, f"{args.model.fit.ckpt_name}.ckpt") # do a concatenatenation with the checkpoint name in neptune
            logger.experiment[f"model/checkpoints/{args.model.fit.ckpt_name}"].download(ckpt_path)
            trainer.fit(model, note_data_module, ckpt_path=ckpt_path)
    else: 
        trainer.fit(model, note_data_module)
 
    # Log objects after `fit` or `test` methods
    # model summary
    #logger.log_model_summary(model=model, max_depth=-1)

    # Predict with model 
    predictions = classify_notes(trainer, model, note_data_module)
    true_labels = np.array(val_df[CLASSES].values.tolist())
    
    # result_folder = os.path.join(ROOT_DIR, args.model.predict.result_folder)
    evaluate_model(true_labels, predictions, DIAGNOSES, logger) # threshold=model_config.model.threshold, result_folder=result_folder, model_name=args.model.model_name
    




if __name__ == "__main__":
    train_model()