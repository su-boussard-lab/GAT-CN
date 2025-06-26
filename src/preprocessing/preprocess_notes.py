# Load packages for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load packages for Big Query 

import nltk.data
import warnings
from tqdm import tqdm
from collections import OrderedDict
from difflib import SequenceMatcher
import re
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
tqdm.pandas()

# For preprocessing
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from negspacy.termsets import termset
#from src.utils.config import config

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

nlp1 = spacy.load("en_core_web_sm")
nlp2 = spacy.load("en_core_sci_sm")
ts = termset("en_clinical")
nlp2.add_pipe(
    "negex",
    config={
        "chunk_prefix": ["no"],
    },
    last=True,
)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def trim_notes(txt, regexp, neighborhood_length = 2, split_method = 'punctuation', with_vocabulary=True): 
    """ Given a note, keeps only the sentences with a mention of depression,
        along with surrounding sentences (neighborhood_length sentences before,
        neighborhood_length sentences after)
        
        Notes can be split into sentences either by regex, or using a nltk 
        tokenizer. This is parametrized by split_method.
        
        Output is a list of sentences """
    
    if split_method == 'punctuation':
        list_sentence = re.split('\.\s|\n|\s\s\s', txt)
    else:
        list_sentence = tokenizer.tokenize(txt)
    
    list_sentence = [sent.strip() for sent in list_sentence if (len(sent.strip())>=2)]
    
    if not with_vocabulary:
        return list_sentence
    
    dep_indicators = [bool(regexp.search(sent.lower())) for sent in list_sentence]
    
    sentences_to_keep_indexes = [z for i,x in enumerate(dep_indicators) 
                             for z in range(i-neighborhood_length,i+neighborhood_length+1) 
                             if x and z >= 0 and z < len(dep_indicators)]
    # remove duplicates 
    sentences_to_keep_indexes = [*set(sentences_to_keep_indexes)]

    trimmed_sentence_list = [list_sentence[i] for i in sentences_to_keep_indexes]
    
    return trimmed_sentence_list

def format_for_save(sent_list):
    return '[SEP]'.join(sent_list)


def similar(a, b):
    """ Returns similarity of sentences a and b (between 0 and 1)"""
    return SequenceMatcher(None, a, b).ratio()

def shorten_similar(sent_list, similarity_threshold = 0.8):
    """
        Given a list of sentences and a similarity threshold, 
        returns a largest subset of sent_list that has no pair
        of sentences with similarity above the threshold.
        
        Complexity: quadratic in length of sentence list
        
        Returns: merged sentences (string)
    """
    cleaned_all_sentences = []
    for p in sent_list:
        if all(similar(p, q) < similarity_threshold for q in cleaned_all_sentences):
            cleaned_all_sentences.append(p)
    merged_sentences = "\n".join(cleaned_all_sentences)
    return merged_sentences

def shorten_similar_df_parallel(df, progress_bar = False):
    """ Applies shorten_similar to a Pandas dataframe.
        progress_bar argument defines the use of tqdm"""
    if progress_bar:
        df['sentence_list'] = df['sentence_list'].progress_apply(shorten_similar)
    else:
        df['sentence_list'] = df['sentence_list'].apply(shorten_similar)
    return df


def aux_cpt_words(sent):
    """ Returns the number of words in a note"""
    try:
        words = [w for w in re.split(" |\n", sent) if (w != " " and w not in "\"'`~!@#$%^&*()-_=+{}[]\\|\t\n,<.>/?:;")]
        return len(words)
    except:
        return np.nan
    

def remove_redundancy(terms):
    terms = [str(x).lower() for x in terms if str(x).lower() != 'nan']
    terms = list(set(terms))

    still_included = True
    while still_included:
        still_included = False
        for y in terms:
            temp = [x for x in terms if y not in x]
            if len(temp) != len(terms) - 1:
                still_included = True
                terms = [y] + temp
                break

    ## Sanity check
    mat = np.zeros((len(terms), len(terms)))
    for i in range(len(terms)):
        for j in range(len(terms)):
            if terms[i] in terms[j]:
                mat[i,j] = 1
    assert(mat.sum() == mat.shape[0])
    
    return terms

def preprocess_text(text: str) -> str:
    """Preprocess text, such that the medical notes look somewhat okayish, and can also be fed to ClinicalBERT
       Requires nlp = spacy.load("en_core_web_sm")
    Args:
        text (str): input text, unprocessed
        threshold (int, 1): threshold for length of a word
    Returns:
        processed_text (str): processed text
    """
    # re.IGNORECASE ignoring cases
    # compilation step to escape the word for all cases
    compiled = re.compile(re.escape("stanford hospital and clinics"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford hospitals and clinics"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford cancer center"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford thoracic oncology"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford health care"), re.IGNORECASE)
    text = compiled.sub("", text)
    compiled = re.compile(re.escape("stanford healthcare"), re.IGNORECASE)
    text = compiled.sub("", text)
    
    # Replace weird character:
    text = text.replace("¿", "")
    text = text.replace("#", "")
    
    # Clinical bert preprocessing 
    text = re.sub("m\.d\.", "md",  text)
    text = re.sub("dr\.", "doctor", text)
    
    # https://www.kaggle.com/code/naim99/clinical-bert-text-classification
    text = re.sub("http\S+", "", text)
    text = re.sub('w{3}\S+', "", text)
   
    return text
    

def shorten(note, regexp, max_length = 512):
    """ Given a note that is too long, returns shortened version of it.
        Step 1: Make the neighborhood length 1 instead of 2 
        Step 2: If step 1 wasn't enough to make the length less than max_length,
                sort the sentences with no depression mention by length, and
                delete those sentences from longest to shortest. Stop when length goes below max_length
        Return note after step 1 and 2 (length could still be > max_length, but much less probable now)
        """
    
    all_sentences = [sentence for paragraph in note.split("\n\n") for sentence in paragraph.split("\n")]
    
    sentences_wth_depression = [bool(regexp.search(x.lower())) for x in all_sentences]
    
    index_positive = [i for i,x in enumerate(sentences_wth_depression) if x] # indices of sentences with mention of depression

    # Number of sentences between sentences with mentions
    steps = [index_positive[i+1] - index_positive[i] for i in range(len(index_positive)-1)]
    
    # We delete sentences that are in the "middle" section of the part between two positive sentences (positive = with depression mention)
    to_delete = [z for (x,y) in zip(steps, index_positive[:-1]) if x > 3 for z in range(y+2,y+x-1)]

    all_sentences = [x for i,x in enumerate(all_sentences) if i not in to_delete]
    
    merged_sentences = "\n".join(all_sentences)
    new_length = len([w for w in re.split(" |\n", merged_sentences) if (w != " " and w not in "\"'`~!@#$%^&*()-_=+{}[]\\|\t\n,<.>/?:;")])
    
    if new_length <= max_length:
        return merged_sentences
    
    else:    
        sentences_wth_depression = [bool(regexp.search(x.lower())) for x in all_sentences]
        
        # Sort false sentences (no mention) by length (in reverse order)
        sort_lengths_falses = sorted([(i,aux_cpt_words(x)) for i,x in enumerate(all_sentences) if sentences_wth_depression[i] == False], key = lambda x: -x[1])
        
        cum_sums = np.cumsum([x[1] for x in sort_lengths_falses]) 
        cum_sums = np.concatenate(([0], cum_sums))

        # find out the index of the last sentence we need to delete
        last_to_delete = np.where(cum_sums < new_length - max_length)[0][-1] + 1
        to_delete = [x[0] for x in sort_lengths_falses[:last_to_delete+2]]


        all_sentences = [x for i,x in enumerate(all_sentences) if i not in to_delete]
            
        merged_sentences = "\n".join(all_sentences)
        new_length = len([w for w in re.split(" |\n", merged_sentences) if (w != " " and w not in "\"'`~!@#$%^&*()-_=+{}[]\\|\t\n,<.>/?:;")])
        
            
        return merged_sentences
    
    
def preprocess(note_outcome, diag_by_patient, terms, folder_name, with_vocabulary=True): 
    terms = remove_redundancy(terms)
    print('Number of vocab terms: {}'.format(len(terms)))

    print(f'Note outcome shapes: {note_outcome.shape}')

    regexp = re.compile('|'.join(terms))
    # Filter to get notes with mentions of diagnoses
    if with_vocabulary: 
        df_notes_with_mentions = note_outcome[note_outcome["note"].progress_apply(lambda x: bool(regexp.search(x.lower())))][['pat_deid', 'note', 'effective_time']].copy(deep=True)
        df_notes_with_mentions = df_notes_with_mentions.drop_duplicates()
        print(f'Df_notes_with_mentions: {df_notes_with_mentions.shape}')
        nb_pat_with_mentions = len(df_notes_with_mentions.groupby(['pat_deid']).count())
        print(f"nb_pat_with_mentions = {nb_pat_with_mentions}")
    else: 
        df_notes_with_mentions = note_outcome[['pat_deid', 'note', 'effective_time']]
    
    ## Trim notes, keeping neighborhoods of diagnoses mentions
    df_notes_with_mentions['sentence_list'] = df_notes_with_mentions.note.progress_apply(lambda x: trim_notes(x, regexp, neighborhood_length = 2, with_vocabulary=with_vocabulary)) # try neighborhood_length = 1

    ## Create patient-level note
    # Group all notes of patients into a single note, represented as list of sentences
    # For that, it is important to first sort the dataframe by effective time of notes, 
    # so that we get a note with sentences in chronological order
    grouped_by_pat = df_notes_with_mentions.sort_values(by = 'effective_time').groupby('pat_deid')[['sentence_list']].agg(lambda x: [y for z in list(x) for y in z]).reset_index()

    # This deletes duplicate sentences, while keeping the order of sentences.
    grouped_by_pat['sentence_list'] = grouped_by_pat['sentence_list'].progress_apply(lambda x: list(OrderedDict.fromkeys(x)))

    # Instead of having a list of sentences, we want to have a string with a clear separator. This makes 
    # processing the csv file later on easier (e.g. when moving from local machine to VM)
    grouped_by_pat['sentence_list'] = grouped_by_pat['sentence_list'].progress_apply(format_for_save)

    # Merge to get outcome (keep patients who have no note)
    grouped_by_pat = grouped_by_pat.merge(diag_by_patient, how = 'right', on = 'pat_deid')
    # Save to csv to parallelize on VM
    grouped_by_pat.to_csv(f'{folder_name}/notes_to_process.csv', index = False)

    ## Trim notes based on similarity: this step is parallelized, so it's much faster on VM!!
    df_notes = pd.read_csv(f'{folder_name}/notes_to_process.csv') 
    df_notes_notna = df_notes.dropna() # Only keep patients who have a note

    # Reverse the saved format to get a list
    df_notes_notna['sentence_list'] = df_notes_notna['sentence_list'].progress_apply(lambda x: x.split('[SEP]'))

    ## Real processing: take into account length to divide work evenly
    
    # Very long to run so run it only if necessary
    df_notes_notna = shorten_similar_df_parallel(df_notes_notna, progress_bar=True)
    # Remerging to df_notes (to keep patients with no note in saved file)
    df_notes = df_notes[["pat_deid", "class_1", "class_2", "class_3", "class_4", "total"]].merge(df_notes_notna[['pat_deid', 'sentence_list']], how = 'left', on = 'pat_deid')
    df_notes.rename(columns = {'sentence_list': 'all_sentences_merged'}, inplace = True)
    df_notes.to_csv(f'{folder_name}/notes_processed.csv', index = False)
    
    ## Notes length distribution 
    df_notes = pd.read_csv(f'{folder_name}/notes_processed.csv')
    
    # Preprocessing 
    df_notes_notna['sentence_list'] = df_notes_notna['sentence_list'].map(lambda x: preprocess_text(x))
    df_notes_notna["n_words"] = df_notes_notna.sentence_list.progress_apply(aux_cpt_words)
    
    print("max:", df_notes_notna["n_words"].max())
    print("min:", df_notes_notna["n_words"].min())
    print("mean:", df_notes_notna["n_words"].mean())
    print("median:", df_notes_notna["n_words"].median())
    print("std:", df_notes_notna["n_words"].std())

    df_notes_notna.hist(column="n_words", bins=100)
    plt.yscale('log')
    plt.savefig(f'{folder_name}/hist_notes_length_distribution.png')

    ##  Shorten the notes that are too long
    long_notes = df_notes_notna[df_notes_notna.n_words > 512].copy(deep = True)
    print(f'len long notes: {len(long_notes)}')

    if with_vocabulary: 
        long_notes['sentence_list'] = long_notes.sentence_list.progress_apply(lambda x: shorten(x, regexp))
    long_notes['new_n_words'] = long_notes.sentence_list.apply(aux_cpt_words)
    print(long_notes.new_n_words.describe())

    long_notes[long_notes.new_n_words > 512].shape[0], long_notes[long_notes.new_n_words > 1024].shape[0]

    n_too_long = long_notes[long_notes.new_n_words > 512].shape[0]
    print('Proportion of notes that exceed max length: {}'.format(100*n_too_long/20000))

    ## This is the histogram of originially long notes, not all notes!
    long_notes.hist(column="new_n_words", bins=100)
    plt.yscale('log')

    plt.savefig(f'{folder_name}/hist_long_notes_length_distribution.png')

    # Final Notes
    final_notes = df_notes.merge(long_notes[['pat_deid', 'sentence_list']], how = 'left', on = 'pat_deid')
    final_notes['sentence_list'] = final_notes['sentence_list'].fillna(final_notes['all_sentences_merged'])
    final_notes = final_notes.drop(columns = ['all_sentences_merged']).rename(columns = {'sentence_list': 'all_sentences_merged'})
    final_notes['n_words'] = final_notes.all_sentences_merged.apply(aux_cpt_words)

    final_notes.hist(column="n_words", bins=100)
    plt.yscale('log')
    plt.savefig(f'{folder_name}/hist_final_notes_length_distribution.png')

    print(f'Final notes with words < 512: {(final_notes.n_words > 512).sum()}') 

    final_notes.to_csv(f'{folder_name}/final_notes_large.csv', index=False)
    print(f'Final notes shape: {final_notes.shape}' )
    
    
