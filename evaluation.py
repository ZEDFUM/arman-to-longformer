import bert_score
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline
arman_tokenizer = AutoTokenizer.from_pretrained("alireza7/ARMAN-MSR-persian-base",model_max_length=512)
arman_summarizer = pipeline("summarization", model="alireza7/ARMAN-MSR-persian-base",tokenizer=arman_tokenizer,device=0)
zedfum_summarizer = pipeline("summarization", model="zedfum/arman-longformer-8k-finetuned-ensani",device=0)
dataset = load_dataset("zedfum/long-summarization-persian",split="test[:10%]").shuffle().select(range(100))

arman_summaries=[]
zedfum_summaries=[]

for doc in tqdm(dataset):
  arman_summary=arman_summarizer(doc['article'], min_length=5,truncation=True, max_length=256)
  zedfum_summary=zedfum_summarizer(doc['article'], min_length=5,truncation=True, max_length=512)
  arman_summaries.append(arman_summary[0]["summary_text"])
  zedfum_summaries.append(zedfum_summary[0]["summary_text"])

# Compute BERTScore
arman_P, arman_R, arman_F1 = bert_score.score(arman_summaries, dataset['summary'], lang="fa")
zedfum_P, zedfum_R, zedfum_F1 = bert_score.score(zedfum_summaries, dataset['summary'], lang="fa")

# Print Arman the results
print(f"Arman Precision: {arman_P.mean():.3f}")
print(f"Arman Recall: {arman_R.mean():.3f}")
print(f"Arman F1 score: {arman_F1.mean():.3f}")
# Print Zedfum the results
print(f"Zedfum Precision: {zedfum_P.mean():.3f}")
print(f"Zedfum Recall: {zedfum_R.mean():.3f}")
print(f"Zedfum F1 score: {zedfum_F1.mean():.3f}")
