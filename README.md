# Tuning the Finetuning with QLoRA
Formerly DollyLlaMA till it became a social media handle for a bunch of random people :D.
Code for finetuning open_llama_7b_v2 model on a refined instruction following datasets with QLoRA


![alt text](https://github.com/avisoori-databricks/Dolly_llama/blob/main/dolly_llama_chilling.png?raw=true)

Datasets used here:


b-mc2/sql-create-context - i.e. WikiSQL and Spider (a ~4500 sample of this with table names) - BSD-3-Clause license and cc-by-4.0 respectively

codeparrot/self-instruct-starcoder - bigscience-openrail-m 

mosaicml/dolly_hhrlhf - commercially permissible license

qwedsacf/grade-school-math-instructions - MIT License

All these datasets have commercially permissible licenses (with differing terms. Please inspect each license to make sure your use case agrees with these terms)


These were downloaded from the Hugging Face Hub cleaned, augmented to conform to the Alpaca intruction response format, deduped before use in finetuning. The total size of the training data set was 72763 examples.

Tips for effective training with QLoRA (primarily avoiding overfitting and ensuring generalization when using LoRA):
1. target_modules should include all linear layers instead of just the attention blocks. This increases training time substantially, but still much faster than full finetuning (highest impact)
2. Opt for a higher rank for the low rank matrices e.g. r=16 vs r=8
3. Upcast the layer norms to float 32 for more stable training
4. Use a memory efficient and stable optimizer such as AdamW

And most importantly, once the model is trained, use contrastive search for generating text. Control the alpha and k parameters to tune the output quality. For performing well with both code generation and coherent non-repetitive text generation, alpha=0.5 and k=4 seemed to perform the best. Merge the adapter layers to the base models to improve inference latency.

Model weights, a blog detailed training guidelines, observations and model performance is on the way.

Inputs follow the Alapaca format: 
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

and in particular, for text to SQL the natural language question has to be formated with a prefix and schema as indicated in the following example:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate SQL query: How many Porsche’s were sold in 2022, given the following schema: car_brands (Brand STRING, Brand_id STRING), car_sales (sales INTEGER, year INTEGER, brand_id STRING)

### Response:
```
The process of building a text to sql system with Flan-T5 is highlighted in the following link:https://github.com/avisoori-databricks/text-to-insights.
Please use a model finetuned using code in this repo instead of the finetuned Flan-T5 model as the results are much better.
