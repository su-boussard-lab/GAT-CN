import pandas as pd
import numpy as np
import os 
import scipy.sparse as sp
import hydra
import sys

from math import log
from split import get_data_and_vocab

# setting path
sys.path.append('../config')
from definitions import ROOT_DIR


# For each word, the list of doc that contains it 
def create_word_doc_list(data, vocab):
    word_doc_list = {}
    for word in vocab: 
        relevant_note = data.NOTE.str.upper().apply(lambda x: any(y in x for y in [word.upper()]))
        word_doc_list[word] = data[relevant_note].index.tolist()
    return word_doc_list


# For each doc, the list of vocab words it contains 
def create_doc_word_list(data, vocab, id_word_map): 
    doc_word_list = {}
    relevant_words = data.NOTE.str.upper().apply(lambda x: any(y in x for y in [word.upper()]) for word in vocab)

    for i, row in relevant_words.iterrows():
        df = pd.DataFrame(row).reset_index()
        idx = df[df[i] == True].index.tolist()
        doc_word_list[i] = list({key: id_word_map.get(key) for key in idx}.values())
    return doc_word_list


'''
Doc word heterogeneous graph
'''
def create_word_window_freq_and_word_pair_count(doc_word_list, word_id_map):
    # word co-occurence with context windows
    window_size = 20
    windows = []

    for doc in doc_word_list.keys():
        #doc_words = aux_cpt_words(note.lower())
        #words = list(set(vocab).intersection(set(doc_words)))
        words = doc_word_list[doc]
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    return word_window_freq, word_pair_count, windows


def create_pmi_matrix(word_window_freq, word_pair_count, windows, vocab, vocab_size):
    row = []
    col = []
    weight = []
    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
        if pmi <= 0:
            continue
        row.append(i) 
        col.append(j)
        weight.append(pmi)
        
        adj = sp.csr_matrix((weight, (row, col)), shape=(vocab_size,  vocab_size))
        
    return adj


def create_doc_word_matrix(word_doc_list, windows, word_id_map, doc_size, vocab_size):
    post_dict_row = []
    post_dict_col = []
    post_dict_weight = []

    num_window = len(windows)

    for word in word_id_map.keys():
        if word in word_doc_list.keys(): 
            docs = word_doc_list[word]
            for doc in docs: 
                i = doc
                j = word_id_map[word]
                post_dict_row.append(i) 
                post_dict_col .append(j)
                post_dict_weight.append(1.0)
                
    adj = sp.csr_matrix((post_dict_weight, (post_dict_row, post_dict_col)), shape=(doc_size, vocab_size))
    return adj


def save_edges_matrix(df, dictionary, folder_name, pmi_matrix=False):
    vocab = dictionary["lexicon"].values.tolist()
    vocab_size = len(vocab)
    
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
    
    id_word_map = {v: k for k, v in word_id_map.items()}
    
    word_doc_list = create_word_doc_list(df, vocab)
    doc_word_list = create_doc_word_list(df, vocab, id_word_map)
    word_window_freq, word_pair_count, windows = create_word_window_freq_and_word_pair_count(doc_word_list, word_id_map)
    
    if pmi_matrix: 
        # create pmi matrix 
        dic_dic_pmi = create_pmi_matrix(word_window_freq, word_pair_count, windows, vocab, vocab_size)
        # np.save("../graph_data/TrainVal/train_dic_dic_pmi.npy", pmi.toarray())
        np.save(f"{folder_name}/dic_dic_pmi.npy", dic_dic_pmi.toarray())
        
    # create_doc_word_matrix
    doc_word_matrix = create_doc_word_matrix(word_doc_list, windows, word_id_map, df.shape[0], vocab_size)
    # np.save("../graph_data/TrainVal/train_word_matrix.npy", doc_word_matrix.toarray())
    np.save(f"{folder_name}/doc_word_matrix.npy", doc_word_matrix.toarray())
    
   

@hydra.main(
    config_path=".././config",
    version_base=None,
    config_name="config_create_edges",
)  
def create_edges_matrix(cfg):
    preprocessed_data_folder = os.path.join(ROOT_DIR, cfg.preprocessed_data_folder)
    graph_data_folder = os.path.join(ROOT_DIR, cfg.graph_data_folder)
    dic_path = os.path.join(ROOT_DIR, cfg.dic_path)
    train_df, val_df, _ = get_data_and_vocab(os.path.join(preprocessed_data_folder, cfg.train_path),
                                             os.path.join(preprocessed_data_folder, cfg.val_path)
                                            )
 
    dictionary = pd.read_pickle(os.path.join(dic_path, "dic_feature.pkl"))
    
    if cfg.pmi_matrix: 
        save_edges_matrix(train_df, dictionary, folder_name=os.path.join(graph_data_folder, cfg.train_path), pmi_matrix=cfg.pmi_matrix)
    save_edges_matrix(val_df, dictionary, folder_name=os.path.join(graph_data_folder, cfg.val_path))
    
  


if __name__ == "__main__":
    create_edges_matrix()

