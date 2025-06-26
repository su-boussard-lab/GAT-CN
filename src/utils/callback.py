import time
from pytorch_lightning.callbacks import Callback

class EpochTimeLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.predict_start_time = None 
        self.training_iter = 1
        self.predict_iter = 1

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() -  self.train_start_time
        trainer.logger.log_metrics({"train_epoch_time": epoch_time}, step=self.training_iter)
        self.training_iter += 1
        
    def on_predict_epoch_start(self, trainer, pl_module):
        self.predict_start_time = time.time()

    def on_predict_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.predict_start_time
        trainer.logger.log_metrics({"predict_epoch_time": epoch_time}, step=self.predict_iter)
        self.predict_iter += 1
        
