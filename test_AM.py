import json
from pathlib import Path

from src.hipporag import HippoRAG

corpus_path = Path("D:/Minh Anh/Custom-HippoRAG/data/rag_qa_test/AM/AM_corpus.json")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]


save_dir = 'outputs'# Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'gpt-4o-mini' # Any OpenAI model name
embedding_model_name = 'text-embedding-3-small'# Embedding model name (NV-Embed, GritLM or Contriever for now)

#Startup a HippoRAG instance
hipporag = HippoRAG(save_dir=save_dir, 
                    llm_model_name=llm_model_name,
                    embedding_model_name=embedding_model_name) 

#Run indexing
hipporag.index(docs=docs)

#Separate Retrieval & QA
samples_path = Path("D:/Minh Anh/Custom-HippoRAG/data/rag_qa_test/AM/AM_closed_end.json")
with open(samples_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

all_queries = [s['question'] for s in samples]

answers = [s['answer'] for s in samples]

gold_docs = []

for sample in samples:
    gold_paragraphs = []

    for item in sample['paragraphs']:
        if item.get('is_supporting') is False:
            continue
        gold_paragraphs.append(item)

    gold_doc = [
        item['title'] + '\n' + item.get('text', item.get('paragraph_text', ''))
        for item in gold_paragraphs
    ]

    gold_docs.append(gold_doc)

rag_results = hipporag.rag_qa(queries=all_queries, 
                              gold_docs=gold_docs,
                              gold_answers=answers)

print("\nRecall@1:", rag_results[3]['Recall@1'])
print("\nRecall@5:", rag_results[3]['Recall@5'])
print("\n", rag_results[4])


