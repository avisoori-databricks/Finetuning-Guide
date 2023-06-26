# Dolly_llama
Code for finetuning openllama models on instruction following datasets with QLoRA


![alt text](https://github.com/avisoori-databricks/Dolly_llama/blob/main/dolly_llama_chilling.png?raw=true)

Datasets used here:

mosaicml/dolly_hhrlhf

b-mc2/sql-create-context

b-mc2/wikihow_lists

These were downloaded from the Hugging Face Hub cleaned, augmented to conform to the Alpaca intruction response format, deduped before use in finetuning.
Model weights, distributed training guidelines and full write up TBD.
