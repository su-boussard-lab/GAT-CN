    
import pandas as pd
import os
import numpy as np
#from bigquery import load_table, save_table
from preprocessing.preprocess_notes import preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
 
# setting path
# sys.path.append('../config')
from config.definitions import ROOT_DIR

import hydra 

def filter_vocab(vocab_per_class, note_outcome):
    single_label_df = note_outcome[note_outcome["total"] == 1]
    note_class_1 = single_label_df[single_label_df["class_1"] == 1.0]
    note_class_2 = single_label_df[single_label_df["class_2"] == 1.0]
    note_class_3 = single_label_df[single_label_df["class_3"] == 1.0]
    note_class_4 = single_label_df[single_label_df["class_4"] == 1.0]

    note_classes = [note_class_1, note_class_2, note_class_3, note_class_4]
    
    total_vocab = pd.DataFrame()
    for i, class_name in enumerate(["class_1", "class_2", "class_3", "class_4"]):
        note_class = note_classes[i]
        vocab = vocab_per_class[vocab_per_class["DIAGNOSE"] == class_name]
        names = vocab["NAME"]
        d = dict()
        for word in names: 
            d[word] = note_class['note'].str.upper().apply(lambda x: any(y in x for y in [word.upper()])).sum()
        vocab["OCCURENCE"] = d.values()
        total_vocab = pd.concat([total_vocab, vocab[vocab["OCCURENCE"] != 0]])
        total_vocab = total_vocab.sort_values(["DIAGNOSE", "OCCURENCE"], ascending=False)

    return total_vocab


def get_dataset(notes, diag_by_patient, pat_ids):
    notes = notes[notes["pat_deid"].isin(pat_ids)]
    diag_by_patient = diag_by_patient[diag_by_patient["pat_deid"].isin(pat_ids)]
    diag_by_patient.sort_values(["pat_deid"], inplace=True) # UTILE?
    # assign notes to their patients classes
    note_outcome = pd.merge(notes, diag_by_patient, on="pat_deid")
    return note_outcome, diag_by_patient


def save_preprocessed_data_and_vocab(train_folder, test_folder, vocab_per_class, with_vocabulary=True):
    # Load all notes
    df_notes = load_table("note_by_type")
    df_notes.columns = [c.lower() for c in df_notes.columns]

    ## Get outcomes for all patients 
    diag_by_patient = load_table("outcome_table_1")
    diag_by_patient.columns = [c.lower() for c in diag_by_patient.columns]
    print('Number of total patients: {}'.format(len(pd.unique(diag_by_patient["pat_deid"]))))

    train_ids = pd.read_csv(f'{train_folder}/ids.csv').PAT_DEID.tolist()
    test_ids = pd.read_csv(f'{test_folder}/ids.csv').PAT_DEID.tolist()

    train_dataset, train_diagnoses = get_dataset(df_notes, diag_by_patient, train_ids)
    test_dataset, test_diagnoses = get_dataset(df_notes, diag_by_patient, test_ids)
    
    # Filter vocab: 
    vocab_per_class = filter_vocab(vocab_per_class, train_dataset)
    vocab_per_class.to_csv(f'{train_folder}/vocab_per_class.csv', index=False)
    
    # Preprocess train dataset
    preprocess(train_dataset, train_diagnoses, vocab_per_class["NAME"], folder_name=train_folder, with_vocabulary=with_vocabulary)
    preprocess(test_dataset, test_diagnoses, vocab_per_class["NAME"], folder_name=test_folder, with_vocabulary=with_vocabulary)

def get_preprocessed_data(folder_name):
    notes_df = pd.read_csv(f'{folder_name}/final_notes_large.csv').dropna()
    notes_df.columns = [c.upper() for c in notes_df.columns]
    notes_df.rename(columns = {'ALL_SENTENCES_MERGED': "NOTE"}, inplace = True)
    notes_df.rename(columns = {'N_WORDS': "WORD_COUNT"}, inplace = True)
    
    return notes_df[["PAT_DEID", "NOTE","CLASS_1", "CLASS_2", "CLASS_3", "CLASS_4"]]


def get_data_and_vocab(train_folder, test_folder):
    train_data_preprocessed = get_preprocessed_data(folder_name=train_folder)
    test_data_preprocessed = get_preprocessed_data(folder_name=test_folder)
    training_vocab = pd.read_csv(f"{train_folder}/vocab_per_class.csv")

    return train_data_preprocessed, test_data_preprocessed, training_vocab


def split(X, y, train_folder, test_folder,  stratify=True, test_size=0.06):   
    if stratify: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42) 
    else: 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) 

    pd.DataFrame(X_train, columns = ["PAT_DEID"])["PAT_DEID"].to_csv(f'{train_folder}/ids.csv', index=False)
    pd.DataFrame(X_test, columns = ["PAT_DEID"])["PAT_DEID"].to_csv(f'{test_folder}/ids.csv', index=False)  
    return  X_train, X_test, y_train, y_test


def get_new_labels(y): 
    le =  LabelEncoder()
    y_new = le.fit_transform([''.join(str(l)) for l in y])
    return y_new, le

@hydra.main(
    config_path=".././config",
    version_base=None,
    config_name="config_split",
)
def create_preprocessed_dataset(cfg):
    classes = ["CLASS_1", "CLASS_2", "CLASS_3", "CLASS_4"]  
    
    data_path = os.path.join(ROOT_DIR, cfg.data_path)

    # Load outcome_table_1 
    outcome_table_1 = load_table("outcome_table_1")
    # Load vocab (created from the full training data (full_X_train below))
    vocab_per_class = load_table("vocab_per_class")
    print(f"Total vocab shape: {vocab_per_class.shape}")
    
    X = np.array(outcome_table_1["PAT_DEID"])
    y = np.array(outcome_table_1[classes])

    # Train/Test split
    full_X_train, X_test, full_y_train, y_test = split(X, y, f'{data_path}/TrainTest/train', f'{data_path}/TrainTest/test', stratify=True, test_size=0.06)
    print(f'Size of the training and test set: {full_X_train.shape[0]}, {X_test.shape[0]}')
    print('Preprocessing train and test set ...')
    save_preprocessed_data_and_vocab(f'{data_path}/TrainTest/train', f'{data_path}/TrainTest/test', vocab_per_class, with_vocabulary=cfg.with_vocabulary)
    print('Preprocessing done for train and test set')

    # Train/Val split
    X_train, X_val, y_train, y_val = split(full_X_train, full_y_train, f'{data_path}/TrainVal/train',f'{data_path}/TrainVal/val',  stratify=True, test_size=0.2)
    print(f'Size of the training and validation set: {X_train.shape[0]}, {X_val.shape[0]}')
    print('Preprocessing train and val set ...')
    save_preprocessed_data_and_vocab(f'{data_path}/TrainVal/train', f'{data_path}/TrainVal/val', vocab_per_class, with_vocabulary=cfg.with_vocabulary)
    print('Preprocessing done for train and val set')
    
    if cfg.cross_val: 
        # CV: 4 folds Train/Val Split 
        n_folds = 4 
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        full_train_ids = pd.DataFrame(full_X_train, columns = ["PAT_DEID"])["PAT_DEID"].tolist()
        full_y_train_new, le = get_new_labels(full_y_train)

        folds = [(np.array(full_train_ids)[train_indexes].tolist(), 
        np.array(full_train_ids)[val_indexes].tolist()) for train_indexes, val_indexes in kf.split(full_train_ids, full_y_train_new)]
        for i, (train_pat_deids, val_pat_deids) in enumerate(folds):
            print(f'Fold {i + 1}: training and validation set size {len(train_pat_deids)}, {len(val_pat_deids)}')
            fold_folder = f"{data_path}/fold_{i+1}"
            pd.DataFrame(train_pat_deids, columns = ["PAT_DEID"])["PAT_DEID"].to_csv(f'{fold_folder}/train/ids.csv', index=False)
            pd.DataFrame(val_pat_deids, columns = ["PAT_DEID"])["PAT_DEID"].to_csv(f'{fold_folder}/val/ids.csv', index=False)  
            print(f'Fold {i + 1}: Preprocessing train and val set ...')
            save_preprocessed_data_and_vocab(f"{fold_folder}/train", f"{fold_folder}/val", vocab_per_class, with_vocabulary=cfg.with_vocabulary)
            print(f'Fold {i + 1}: Preprocessing done for train and val set')
    

if __name__ == "__main__":
    create_preprocessed_dataset()