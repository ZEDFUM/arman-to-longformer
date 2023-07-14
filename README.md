# arman-to-longformer
This project use Longformer's attention mechanism to [alireza7/ARMAN-MSR-persian-base](https://huggingface.co/alireza7/ARMAN-MSR-persian-base) in order to perform abstractive summarization on long documents. so new model can accept 8K tokens (rather than 512 tokens).it should be fine-tuned for summarization tasks.

new model is available in [huggingface](https://huggingface.co/zedfum/arman-longformer-8k-finetuned-ensani)



## ⚡️ Quickstart
```
from transformers import AutoTokenizer
from transformers import pipeline

summarizer = pipeline("summarization", model="zedfum/arman-longformer-8k-finetuned-ensani", tokenizer="zedfum/arman-longformer-8k-finetuned-ensani" , device=0)
text_to_summarize=""
summarizer(text_to_summarize, min_length=5, max_length=512)
```
