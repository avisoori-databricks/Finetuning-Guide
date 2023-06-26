# Databricks notebook source
# MAGIC %pip install trl
# MAGIC %pip install git+https://github.com/huggingface/peft.git
# MAGIC %pip install bitsandbytes

# COMMAND ----------

# MAGIC %sql
# MAGIC USE dolly_llama;

# COMMAND ----------

df = spark.sql("SELECT * FROM dolly_llama_instruct").toPandas()
df['text'] = df["prompt"]+'\n'+df["response"]
df.drop(columns=['prompt', 'response'], inplace=True)
display(df)

# COMMAND ----------

from datasets import load_dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.05)

# COMMAND ----------

from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from transformers.trainer_callback import TrainerCallback
import os
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
import mlflow

# COMMAND ----------

mlflow.set_experiment(f"/Users/avinash.sooriyarachchi@databricks.com/Dolly_Llama-7b_auto")

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
            
callbacks = [PeftSavingCallback()]

# COMMAND ----------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules = ["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

base_dir = "/dbfs/FileStore/shared_uploads/avinash.sooriyarachchi@databricks.com/<your_subdirectory>/"

per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = 'adamw_bnb_8bit'
learning_rate = 1e-5
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "linear"


# COMMAND ----------

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=base_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs = 10,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

# COMMAND ----------

nf4_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)

# COMMAND ----------

model_path = 'openlm-research/open_llama_7b'

# COMMAND ----------

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({"additional_special_tokens": ["\n### End", "### Instruction:", "### Response:\n"]})
#Becuase https://github.com/huggingface/transformers/issues/22312
tokenizer.pad_token_id = (
        0  
    )
tokenizer.padding_side = "left"  # Allow batched inference

# COMMAND ----------

model = LlamaForCausalLM.from_pretrained(
    model_path, quantization_config=nf4_config, device_map='auto',
)

# COMMAND ----------

trainer = SFTTrainer(
    model,
    peft_config=lora_config,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
    callbacks=callbacks
)
#Pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
  if "norm" in name:
    module = module.to(torch.float32)


# COMMAND ----------

trainer.train()