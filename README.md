# DollyLlama-7B
Code for finetuning openllama7bv2 model on instruction following datasets with QLoRA


![alt text](https://github.com/avisoori-databricks/Dolly_llama/blob/main/dolly_llama_chilling.png?raw=true)

Datasets used here:


b-mc2/sql-create-context - i.e. WikiSQL and Spider (a ~4500 sample of this with table names) - BSD-3-Clause license and cc-by-4.0 respectively

codeAlpaca-20k - cc-by-4.0 license according to repo: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k

mosaicml/dolly_hhrlhf - commercially permissible licesne

qwedsacf/grade-school-math-instructions - MIT License

All these datasets have commercially permissible licenses


These were downloaded from the Hugging Face Hub cleaned, augmented to conform to the Alpaca intruction response format, deduped before use in finetuning. The total size of the training data set was 72763 examples.

Tips for effective training with QLoRA (primarily avoiding overfitting and ensuring generalization when using LoRA):
1. target_modules should include all linear layers instead of just the attention blocks. This increases training time substantially, but still much faster than full finetuning
2. Opt for a higher rank for the low rank matrices e.g. r=16 vs r=8
3. Upcast the layer norms to float 32 for more stable training
4. Use a memory efficient and stable optimizer such as ADAMW

And most importantly, once the model is trained use contrastive search for generating text. Control the alpha and k parameters to tune the output quality. For performing well with both code generation and coherent non-repetitive text generation alpha=0.5 and k=4 seemed to perform the best.

Model weights, and a blog detailed training guidelines, observations and model performance is on the way.
