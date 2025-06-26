
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn 

from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from neptune.new.types import File
from config.definitions import DIAGNOSES


def evaluate_model(y_true, y_pred, classes, logger, threshold=0.5, result_folder="results", model_name="ClinicalBERT"):
    
    y_pred_binary = np.where(np.array(y_pred) > threshold, 1., 0.)

    # Confusion matrix for each class
    d = {"Class": classes, "TN":[], "FP":[], "FN":[], "TP":[]}
    for i in range(len(classes)): 
        confusion_matrix  = multilabel_confusion_matrix(y_true, y_pred_binary)[i]
        (tn, fp, fn, tp) = confusion_matrix.ravel()
        d["TN"].append(tn)
        d["FP"].append(fp)
        d["FN"].append(fn)
        d["TP"].append(tp)
    metrics = pd.DataFrame.from_dict(d)
    logger.experiment[f"{result_folder}/metric"].upload(File.as_html(metrics))

    # Classification report 
    report = classification_report(
        y_true,
        y_pred_binary,
        output_dict=True,
        target_names=classes
    ) 
    report = pd.DataFrame.from_dict(report)
    logger.experiment[f"{result_folder}/report"].upload(File.as_html(report))
    # Hamming Loss
    logger.experiment[f"{result_folder}/hamming_loss"] = hamming_loss(y_true, y_pred_binary)
    # Subset Accuracy
    logger.experiment[f"{result_folder}/subset_accuracy"] = accuracy_score(y_true, y_pred_binary)
    
    # AUC - ROC
    f, ax = plt.subplots(figsize=(15, 8))
    for i, attribute in enumerate(DIAGNOSES):
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y_true[:,i].astype(int), y_pred[:, i])
        auc = sklearn.metrics.roc_auc_score(
            y_true[:,i].astype(int), y_pred[:, i])
        ax.plot(fpr, tpr, label='%s %g' % (attribute, auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Trained on Clinical Notes - AUC ROC') 
    logger.experiment[f"{result_folder}/auc_roc.png"].log(File.as_image(f))
    plt.show()
    

