# Databricks notebook source
#Datasets used here:
# mosaicml/dolly_hhrlhf
# b-mc2/sql-create-context
# b-mc2/wikihow_lists
## competition_math - not using because it's in HELM
## qwedsacf/grade-school-math-instructions - not using because it's in HELM

# COMMAND ----------

from datasets import load_dataset , Dataset, concatenate_datasets 
import numpy as np
import pandas as pd
import random

# COMMAND ----------

drhlf = load_dataset("mosaicml/dolly_hhrlhf")
drlhf_train_df = pd.DataFrame(drhlf['train'])
drlhf_test_df = pd.DataFrame(drhlf['test'])
drlhf_df = pd.concat([drlhf_train_df, drlhf_test_df])
display(drlhf_df)

# COMMAND ----------

text_to_sql = load_dataset("b-mc2/sql-create-context")
txsql_df = pd.DataFrame(text_to_sql)
display(txsql_df)

# COMMAND ----------

import json 
dsdf_ = pd.json_normalize(txsql_df['train'])
display(dsdf_)

# COMMAND ----------

dsdf_['question'] = 'Generate SQL query: '+ dsdf_['question']
dsdf_['context'] = dsdf_['context'].apply(lambda x: x.replace("CREATE TABLE", "").replace("VARCHAR", "STRING").strip())
dsdf_['Instruction'] = dsdf_['question']+ ', given the following schema: ' + dsdf_['context']
dsdf_.drop(columns=['question', 'context'], inplace=True)
display(dsdf_)

# COMMAND ----------

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:"""

# COMMAND ----------

dsdf_['prompt'] = dsdf_["Instruction"].apply(lambda x: template.format(x))

# COMMAND ----------

display(dsdf_)

# COMMAND ----------

dsdf_.drop(columns = ['Instruction'], inplace=True)
dsdf_.rename(columns={'answer': 'response'}, inplace=True)
dsdf_ = dsdf_[['prompt', 'response']]
display(dsdf_)

# COMMAND ----------

wikihowds = load_dataset("b-mc2/wikihow_lists")
# drlhf_train_df = pd.DataFrame(drhlf['train'])
# drlhf_test_df = pd.DataFrame(drhlf['test'])
# drlhf_df = pd.concat([drlhf_train_df, drlhf_test_df])
# display(drlhf_df)
wikihowds

# COMMAND ----------

wikihowdf = pd.DataFrame(wikihowds)
wikihowdf = pd.json_normalize(wikihowdf['train'])
wikihowdf.drop(columns=['pageid'], inplace=True)
display(wikihowdf)

# COMMAND ----------

set(wikihowdf['result_type'].to_list())

# COMMAND ----------

#Give ingredients to ......
#What are the items needed to .....
#Give a stepped summary of ......
#Format the data like this and make and instruction following dataset 
# Prepend strings based on result_type
wikihowdf['title'] = wikihowdf.apply(lambda row: 'Give ingredients required to ' + row['title'] if row['result_type'] == 'ingredients' else
                                   'What are the items needed to ' + row['title'] if row['result_type'] == 'needed_items' else
                                   'Give a stepped summary of how to ' + row['title'], axis=1)
wikihowdf.drop(columns=['result_type'], inplace=True)
wikihowdf.rename(columns={'title':'prompt', 'result':'response'}, inplace=True)
display(wikihowdf)

# COMMAND ----------

wikihowdf['prompt'] = wikihowdf["prompt"].apply(lambda x: template.format(x))

# COMMAND ----------

display(wikihowdf)

# COMMAND ----------

dolly_llama_df = pd.concat([drlhf_df, dsdf_, wikihowdf], ignore_index=True).sample(frac=1, random_state=42)

# COMMAND ----------

dolly_llama_df['response'] = dolly_llama_df['response'] +  "\n### End"

# COMMAND ----------

dolly_llama_df.drop_duplicates(inplace=True)
display(dolly_llama_df), dolly_llama_df.shape

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS dolly_llama;
# MAGIC USE dolly_llama;

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE dolly_llama_instruct;

# COMMAND ----------

spark.createDataFrame(dolly_llama_df).write.saveAsTable('dolly_llama_instruct')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dolly_llama_instruct