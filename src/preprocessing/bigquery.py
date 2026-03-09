# Load packages for data analysis
import pandas as pd

# Load packages for Big Query 
from google.cloud import bigquery
import os

# Define configurations for Big Query - Stride Datalake
project_id = 'som-nero-phi-boussard' # Location of stride datalake
es = "som-nero-phi-boussard.ES_ACU_Oncology"
client = bigquery.Client(project=project_id) # Set project to project_id
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/eliasaquand/.config/gcloud/application_default_credentials.json'
os.environ['GCLOUD_PROJECT'] = "som-nero-phi-boussard" # specify environment
db = "som-nero-phi-boussard" # Define the database

# 1) Specify the job config to properly read the file
job_config = bigquery.LoadJobConfig()
job_config.autodetect = True # determines the datatype of the variable
job_config.write_disposition = 'WRITE_TRUNCATE'
job_config.max_bad_records = 1 # allow 5 bad records; 

# Read schema from JSON
# job_config.schema = self.bq_client.schema_from_json(
# f"{json_schema_dir}/{custom_mapping_table}.json")

# 2) Specify destination
# destination = f"som-nero-phi-boussard.MSc_ACU_Oncology.[COHORT NAME]"

# 3) Save file ob Big Query, using result from so far; client is specified above - implemented in the file 
# load_job = client.load_table_from_dataframe(dataframe = chemo_tx_dd,                                  
#                                                     destination = destination,
#                                                     job_config = job_config)

# Run the job:
# load_job.result()

def load_table(table_name: str) -> pd.DataFrame:
    # fetch the table 
    sql_query = f""" SELECT * FROM {es}.{table_name}"""
    table = (client.query(sql_query)).to_dataframe()

    return table 

def save_table(table: pd.DataFrame, table_name: str) -> None: 
    # Specify destination for storing dataframe
    destination = f"som-nero-phi-boussard.ES_ACU_Oncology.{table_name}"

    # Save file to Big Query
    load_job = client.load_table_from_dataframe(dataframe = table,                                  
                                                   destination = destination,
                                                   job_config = job_config)

    # Run the job:
    load_job.result()