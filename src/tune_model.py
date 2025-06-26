
import numpy as np
import timeit
import os
import torch.multiprocessing
import pytorch_lightning as pl
from utils.utils import update_nested_dict
from pytorch_lightning.loggers import NeptuneLogger
import torch
import hydra 
from omegaconf import DictConfig, OmegaConf 
from data.datamodule import NoteBertDataModule, NoteBertGnnDataModule
from model.model import NoteBertClassifier,  NoteBertGnnClassifier
from preprocessing.split import get_data_and_vocab
from config.definitions import ROOT_DIR, CLASSES
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.integration.pytorch_lightning import TuneReportCallback


import sys
sys.path.insert(0, '../preprocessing')

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system') # see if it corrects the issue by resuming the training 


# method to convert list of comments into predictions for each comment
def classify_notes(trainer, model, dm):
   predictions = trainer.predict(model, datamodule=dm)
   flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
   return flattened_predictions 


def train_model(tune_config, checkpoint_dir=None, cfg: DictConfig = None) -> None:
    args = cfg.args 
    
    model_config_updated = OmegaConf.to_container(cfg.model_config, resolve=True)
    update_nested_dict(model_config_updated, tune_config)
    model_config = DictConfig(model_config_updated)
   
    # Logger
    run_id = args.logger["run_id"] if args.model.fit.resume_training else None
    logger = NeptuneLogger( 
        project=args.logger.project_name,  
        with_id=run_id,
        api_token=args.logger.api_token,
        description=args.logger.description,
        prefix=args.logger.prefix,
    )  
    
    # Data 
    preprocessed_data_folder = os.path.join(ROOT_DIR, model_config.data.preprocessed_data_folder)
    if args.data.testing: 
        train_df, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.full_train_path), os.path.join(preprocessed_data_folder, model_config.data.test_path))
    else: 
        train_df, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.train_path), os.path.join(preprocessed_data_folder, model_config.data.val_path))
    model_config.model.train_size = len(train_df) 
        
    if "gnn" in model_config.model.keys():  
        graph_data_folder = os.path.join(ROOT_DIR, model_config.data.graph_data_folder)
        if args.data.testing: 
            train_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.full_train_path}/doc_word_matrix.npy")) 
            val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.test_path}")) 
        else: 
            train_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.train_path}/doc_word_matrix.npy")) 
            val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.val_path}/doc_word_matrix.npy")) 
        
        note_data_module = NoteBertGnnDataModule(train_df, 
                                        val_df, 
                                        train_adj_dict, 
                                        val_adj_dict,
                                        classes=CLASSES, 
                                        batch_size=model_config.model.batch_size, 
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens,
                                        num_workers=model_config.data.num_workers
                                        )
        
        model = NoteBertGnnClassifier(model_config=model_config.model)
        model.preprocess_dataframe(data=model_config.data)
    else: 
        note_data_module = NoteBertDataModule(train_df, 
                                        val_df, 
                                        classes=CLASSES, 
                                        batch_size=model_config.model.batch_size, 
                                        max_token_length=model_config.data.max_token_length,
                                        model_name=model_config.model.bert.model_name,
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens, 
                                        num_workers=model_config.data.num_workers
                                        )
        model = NoteBertClassifier(model_config.model, classes=CLASSES)
                           
    print(f'Size of the training and test data: {train_df.shape}, {val_df.shape}')
    
    ## Train model
    # Datamodule
    note_data_module.setup()

    # Define callbacks
    callbacks = []
    tune_callback = TuneReportCallback(
        metrics={
            "loss": "val_loss",
            #"mean_accuracy": "ptl/val_accuracy"
        },
        on="validation_end")
        
    callbacks.append(tune_callback)
    
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
        deterministic=False, # ensure full reproducibility from run to run you need to set seeds for pseudo-random generators,
        limit_train_batches=model_config.train.limit_train_batches,
        auto_lr_find=model_config.train.tune.auto_lr_find, 
        auto_scale_batch_size=model_config.train.tune.auto_scale_batch_size,
        num_sanity_val_steps=model_config.train.validate.num_sanity_val_steps, 
        check_val_every_n_epoch=model_config.train.validate.check_val_every_n_epoch,
        val_check_interval=model_config.train.validate.val_check_interval,
        limit_val_batches=model_config.train.validate.limit_val_batches,
        resume_from_checkpoint = os.path.join(checkpoint_dir, "checkpoint") if checkpoint_dir else None,
    )
    
    trainer.fit(model, note_data_module)
 

@hydra.main(
    config_path="./config",
    version_base=None,
    config_name="config_train",
)
def tune_model(cfg: DictConfig):
    num_samples=2
    num_epochs=cfg.model_config.model.n_epochs 
    gpus_per_trial=0 

    # tune num of epochs? 
    # tune swarmup? 
    
    initial_params = [
        {"lr": 2e-5, "batch_size": 32,  "dropout_p": 0.5, "weight_decay": 0.001},
       # {"width": 4, "height": 2, "activation": "tanh"},
    ]
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    
    search_config = {
        "lr": tune.choice([2e-5, 5e-5]),#tune.choice([2e-5, 3e-5, 4e-5, 5e-5]), #tune.loguniform(2e-5, 5e-5), # choice? 
        "batch_size": tune.choice([32, 64]), #tune.choice([32, 64, 128]), # add 4, 8, 16?
        "dropout_p": tune.choice([0.2, 0.5]), #tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "weight_decay": tune.choice([0.001, 0.01]),  #tune.choice([0, 0.001, 0.01, 0.1]),   
    }
    
    reporter = CLIReporter(
        parameter_columns=search_config.keys(),
        metric_columns=["loss", "training_iteration"]
    )
    
    train_fn_with_parameters = tune.with_parameters(train_model, cfg=cfg)
    resources_per_trial = {"cpu": 10, "gpu": gpus_per_trial}

    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_clinicalBERT_truncated",
            progress_reporter=reporter,
        ),
        param_space=search_config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config) #log neptune
    

if __name__ == "__main__":
    start = timeit.timeit()
    tune_model()
    end = timeit.timeit()
    print(f"Time to tune:{end - start}")