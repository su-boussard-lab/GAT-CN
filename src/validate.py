
import numpy as np
import os
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
import torch
import hydra 
from omegaconf import DictConfig, OmegaConf 
from data.datamodule import NoteBertDataModule, NoteBertGnnDataModule
from model.model import NoteBertClassifier,  NoteBertGnnClassifier
from utils.metrics import evaluate_model
from preprocessing.split import get_data_and_vocab
from config.definitions import ROOT_DIR, DIAGNOSES, CLASSES
from utils.utils import tmpdir_manager, update_nested_dict
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, '../preprocessing')

RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED)
torch.multiprocessing.set_sharing_strategy('file_system') # see if it corrects the issue by resuming the training 


# method to convert list of comments into predictions for each comment
def classify_notes(trainer, model, dm, ckpt_path):
   predictions = trainer.predict(model, datamodule=dm, ckpt_path=ckpt_path)
   flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
   return flattened_predictions 

def classify_notes(trainer, model, dm, ckpt_path):
   predictions = trainer.predict(model, datamodule=dm, ckpt_path=ckpt_path)
   flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
   return flattened_predictions 

@hydra.main(
    config_path="./config",
    version_base=None,
    config_name="config_validate",
)
def validate_model(cfg: DictConfig) -> None:
    args = cfg.args
    
    logger = NeptuneLogger( 
        project=args.logger.project_name,  
        with_id=args.logger.run_id,
        api_token=args.logger.api_token,
        description=args.logger.description,
        prefix=args.logger.prefix,
    )  
    
    model_config = cfg.model_config
    try: 
        model_config_updated = OmegaConf.to_container(cfg.model_config, resolve=True)
        #update_config = logger.experiment["hyperparams/model_config"].fetch()
        update_nested_dict(model_config_updated, logger.experiment["hyperparams/model_config"].fetch())
        model_config = DictConfig(model_config_updated)
    except: 
        print("No previous config logged")  
        
    # Data 
    preprocessed_data_folder = os.path.join(ROOT_DIR, model_config.data.preprocessed_data_folder)
    if args.data.testing: 
        _, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.full_train_path), os.path.join(preprocessed_data_folder, model_config.data.test_path))
    else: 
        _, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, model_config.data.train_path), os.path.join(preprocessed_data_folder, model_config.data.val_path)) 
    
    
    # # HACK Small - for test
    if args.data.hack: 
        val_df = val_df[:3]

           
    if "gnn" in model_config.model.keys():  
        graph_data_folder = os.path.join(ROOT_DIR, model_config.data.graph_data_folder)
        if args.data.testing: 
            dic_path = os.path.join(graph_data_folder, model_config.data.full_train_path) 
            val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.test_path}/doc_word_matrix.npy")) 
        else: 
            dic_path = os.path.join(graph_data_folder, model_config.data.train_path)
            val_adj_dict = np.load(os.path.join(graph_data_folder, f"{model_config.data.val_path}/doc_word_matrix.npy")) 
        
        note_data_module = NoteBertGnnDataModule(None, 
                                        val_df, 
                                        None, 
                                        val_adj_dict,
                                        classes=CLASSES, 
                                        batch_size=model_config.model.batch_size, 
                                        max_token_length=model_config.data.max_token_length,
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens,
                                        num_workers=model_config.data.num_workers
                                        )
        
        model = NoteBertGnnClassifier(model_config=model_config.model)
        model.preprocess_dataframe(data=model_config.data, dic_path=dic_path)
    else: 
        # For emails: 
        #val_df = load_table("msg_outcome")
        #val_df.rename(columns = {'MESSAGE': "NOTE"}, inplace = True)
        
        note_data_module = NoteBertDataModule(None, 
                                        val_df, 
                                        classes=CLASSES,
                                        batch_size=model_config.model.batch_size, 
                                        max_token_length=model_config.data.max_token_length,
                                        model_name=model_config.model.bert.model_name,
                                        return_overflowing_tokens=model_config.data.return_overflowing_tokens, 
                                        num_workers=model_config.data.num_workers
                                        )

        # model 
        model = NoteBertClassifier(model_config.model, classes=CLASSES)
        
    print(f'Size of the test data: {val_df.shape}')
    
    # get the annoations examples 
    #test_ex = [2000451, 1571524, 2126079, 1733954, 2689548, 63142]
    #val_df = val_df[val_df["PAT_DEID"].isin(test_ex)]
    #np.save(f"{args.model.predict.result_folder}/pat_ids.npy", val_df.PAT_DEID)
    
    
    ## Train model
    # Datamodule
    note_data_module.setup(stage="predict")

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
            mode=model_config.train.callback.checkpointing.mode
        )
        callbacks.append(checkpoint_callback)
    if model_config.train.callback.early_stopping_callback:
        early_stopping_callback = EarlyStopping(monitor=model_config.train.callback.early_stopping.monitor,  patience=model_config.train.callback.early_stopping.patience) 
        callbacks.append(early_stopping_callback)

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
        limit_train_batches=model_config.train.limit_train_batches,
        auto_lr_find=model_config.train.tune.auto_lr_find, 
        auto_scale_batch_size=model_config.train.tune.auto_scale_batch_size,
        num_sanity_val_steps=model_config.train.validate.num_sanity_val_steps, 
        check_val_every_n_epoch=model_config.train.validate.check_val_every_n_epoch,
        val_check_interval=model_config.train.validate.val_check_interval,
        limit_val_batches=model_config.train.validate.limit_val_batches,
    )
    
    ## Predict with model 
    with tmpdir_manager(base_dir="/tmp") as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, f"{args.model.predict.ckpt_name}.ckpt") # do a concatenatenation with the checkpoint name in neptune
        logger.experiment[f"model/checkpoints/{args.model.predict.ckpt_name}"].download(ckpt_path)
        #ckpt_path = os.path.join(f"checkpoints/{args.model.predict.ckpt_name}.ckpt")
        
        predictions = classify_notes(trainer, model, note_data_module, ckpt_path)
        #np.save(f"{args.model.predict.result_folder}/predictions_proba.npy", predictions)
        
        true_labels = np.array(val_df[CLASSES].values.tolist())
        #np.save(f"{args.model.predict.result_folder}/true_labels.npy", true_labels)
    
        predictions_binarized = np.where(np.array(predictions) > model_config.model.threshold, 1., 0.)
        #np.save(f"{args.model.predict.result_folder}/predictions_binarized.npy", predictions_binarized )

        #result_folder = os.path.join(ROOT_DIR, args.model.predict.result_folder)
        evaluate_model(true_labels, predictions_binarized, DIAGNOSES, logger) # threshold=model_config.model.threshold, result_folder=result_folder, model_name=args.model.model_name
        
        model = NoteBertClassifier.load_from_checkpoint(ckpt_path, config=model_config.model, classes=CLASSES)
        text = val_df.iloc[1].NOTE

        def predictor(texts):
            results = []
            for text in texts:
                tokens = note_data_module.val_dataset.tokenize_note(text)
                input = {'input_ids': tokens.input_ids, 'attention_mask': tokens.attention_mask, 'labels': None}
                outputs = model(input)

                #probas = F.sigmoid(outputs[1]).detach()
                probas = torch.sigmoid(outputs[1]).detach().numpy()
                results.extend(probas)
            return np.array(results)
        
        explainer = LimeTextExplainer(class_names=CLASSES)
        
        exp = explainer.explain_instance(text, predictor, num_features=20, labels = [0,1,2,3], num_samples=10000) 
        fig_1 = exp.as_pyplot_figure(label=0)
        plt.savefig("exp_class_1.png")
        fig_2 = exp.as_pyplot_figure(label=1)
        plt.savefig("exp_class_2.png")
        fig_3 = exp.as_pyplot_figure(label=2)
        plt.savefig("exp_class_3.png")
        fig_3 = exp.as_pyplot_figure(label=3)
        plt.savefig("exp_class_4.png")
        plt.show()





if __name__ == "__main__":
    validate_model()