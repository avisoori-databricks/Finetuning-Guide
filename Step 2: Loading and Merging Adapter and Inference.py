# Databricks notebook source
# MAGIC %pip install trl
# MAGIC %pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install bitsandbytes

# COMMAND ----------

import torch
from datasets import load_dataset, Dataset 
import os
import sentencepiece
import pandas as pd
from peft import PeftModel, PeftConfig
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["\n### End", "### Instruction:", "### Response:\n"]})
#https://github.com/huggingface/transformers/issues/22312
tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
tokenizer.padding_side = "left"  # Allow batched inference

# COMMAND ----------

peft_model_id = "openlm-research/open_llama_7b"
base_dir = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/<trained_adapter_path>/"
peft_model_id =base_dir+ peft_model_id

# COMMAND ----------

config = PeftConfig.from_pretrained(peft_model_id)

# COMMAND ----------

device ='cuda'

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    config.base_model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
model = PeftModel.from_pretrained(model, peft_model_id)

# COMMAND ----------

your_question = ......

# COMMAND ----------

task = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}


### Response:""".format(your_question)

# COMMAND ----------

encoding = tokenizer(task, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Merging model weights so as to avoid inference latency  

# COMMAND ----------

model = model.merge_and_unload()