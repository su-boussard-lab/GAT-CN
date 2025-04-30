<img src="boussardlab.png" width="50%" style="margin-right: 20px; margin-top: 10px;" />

----------------------------

# NLP to Detect Adverse Outcomes of ACU Patients Following Chemotherapy Through Clinical Notes 

-----------------------------
We introduce a novel Natural Language Processing (NLP) model, Graph-Augmented Transformer for Clinical Notes (GAT-CN), that combines transformer-based architecture with a graph neural network (GNN), to embed and analyze unstructured clinical data. GAT-CN was designed to extract post-chemotherapy-related diagnoses from clinical notes—such pain, nausea, vomiting, dehydration, diarrhea, anemia, neutropenia, fever, sepsis, and pneumonia.

## Objectives
**First Objective: main question**

    1. Can we classify chemotherapy patients' adverse outcomes through clinical notes?

**Second Objective: quantity + type**

    1. What types of notes? 
    2. How many notes do we need to classify? 
    3. Which of my notes (timing) are the ones that tell me what's going on? e.g do we need notes in between ACU events or only ACU event notes? can we 4. classify with the first half of the notes only? are the notes after chemotherapy most relevant? are the notes close to ACU most relevant? If we can score them somehow (like feature selection) that will be super relevant. 

**Third Objective: edge cases**

    1. Which conditions are difficult to classify?
    2. Are there any mismatch between notes and the diagnosis code (after classification)?

----------------------------

## Key Features
**Population:** Cancer patients having undergone their first chemotherapy and needed ACU (ED visit or hospitalization) within the 30 days follow-up and diagnosed with one of the 10 OP-35 diagnoses.

**Data:** Clinical Notes (ED Provider Notes for ED patients and H&P and Discharge Summaries for Inpatient patients)

**Task:** Multi-label multi-class classification task 

**Classes:** 
- > Class 1: Pain 
- > Class 2: Nausea, Vomiting, Dehydration, Diarrhea 
- > Class 3: Anemia 
- > Class 4: Sepsis, Neutropenia, Fever, Pneumonia

## Development
- Programming Language: Python 
- Computing Platform: Carina (https://carinadocs.sites.stanford.edu)
- Experiment tracker and model registry: Neptune AI (https://neptune.ai/)

## Usage

### Requirements
To clone and run this project, you will need::
- Python version 3.9.15
- PyTorch 1.13.1
- PyTorch Lightning 1.9.0
- Hydra 1.3.2

### Installation 
**Create conda environment** 
1. Create the environment from the environment.yml file: `conda env create -f environment.yml`. The first line of the yml file sets the new environment's name, _env_ here.
2. Activate the new environment: `conda activate env`
3. Verify that the new environment was installed correctly: `conda env list`.

Ressources: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file


### How to Run
**Notebook**
- 1. Cohort Extraction: apply inclusion/exclusion criteria to obtain our target cohort
- 2. Note Types Selection: filter patients clinical notes according to their type 
- 3. Build Outcome Table: extract patient labels 
 
**Data:**
> `preprocess_notes.py`: clinical notes preprocessing 
> `split.py`: split dataset into Train/Test/Val sets and preprocess the notes of each split 
> `create_edges.py`: create adjancy matrices for words-words edges and words-notes edges

**Where to find the data (only available for Stanford community who included in the IRB):**
- data_preprocessed: dataset splits with patients  preprocessed notes 
- graph_data: GNN data input splitted with dictionary embeddings, pmi matrix and word-notes adjancy matrix. 

**Models:** 
There are two main scripts: 
- `train.py`: to train a model 
- `validate.py`: to validate a model from a checkpoint

In this project, we experimented with two types of models BERT and combining BERT with GNN (Graph Neural Networks):
- To train a BERT model: `python train.py model_config=config`
- To train a BERT + GNN model: `python train.py model_config=config_gnn`
- To validate a BERT model: `python validate.py model_config=config`
- To validate a BERT + GNN model: `python validate.py model_config=config_gnn`
By default, the model is BERT i.e `model_config=config` in the config files. 

You can easily tune hyperparameters of the model by changing *model_config*: 
`python train.py model_config.model.n_epochs=5 model_config.model.bert.model_name="distilroberta-base" model_config.model.optimizer.lr: 3e-5`

By default, BERT see the truncated version of the patient preprocessed note i.e only the first `max_token_length=512` tokens. If you want BERT to see the full notes of your patients: 
`python train.py model_config.data.max_token_length=256 model_config.data.return_overflowing_tokens=true`

Neptune AI is used for logging: you can easily set up an account at https://neptune.ai and create a project. Then you can train the model by using your neptune AI logger: 
`python train.py args.logger.project_name=<your_project_name> args.logger.api_token=<your_api_token>`

Neptune AI stores model hyperparameters, resuls, checkpoints, etc but you can also save the model checkpoints by creating a checkpoint folder and run: 
`python train.py model_config.train.callback.checkpointing.dirpath=<your_checkpoint_folder>`

Note: cross_val.py script is outdated. 

## License
MIT

## Authors
Elia Saquand, Behzad Naderalvojoud, Max Schuessler, Malvika Pillai, Brian Travis Rice, Doug Blayney, Tina Hernandez-Boussard



