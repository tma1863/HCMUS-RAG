from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import numpy as np
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import pandas as pd

load_dotenv()

if __name__ == "__main__":
    # Use getpass to securely input the API key

    # Initialize OpenAI embeddings
    openai_embeddings = OpenAIEmbeddings()
    pca = PCA(n_components=3)
    llm = OpenAI()

    num_test = 1
    retrieval_k = 2  # Number of documents to retrieve for each query
    # datasets = ["DS", "MCS", "AM"]
    datasets = ["DS"]
    for major in datasets:
        corpus_path = f"data/test/{major}/{major}_corpus.json"
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        docs = [f"Course ID: {doc['title']}\n{doc['text']}" for doc in corpus]
        content = "; ".join(docs)
        # Create a text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=170, chunk_overlap=0)

        # Create a Document object with content
        chunks = text_splitter.split_documents([Document(page_content=content)])
        # print(chunks)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = openai_embeddings.embed_documents(texts)

        # Convert the list of lists to a NumPy array
        embeddings_array = np.array(embeddings)
        reduced_embeddings = pca.fit_transform(embeddings_array)
        # Create a list of tuples for texts and their embeddings
        text_embeddings = list(zip(texts, embeddings))

        # Store the embeddings in a FAISS vector store using the pre-computed embeddings
        vector_store = FAISS.from_embeddings(text_embeddings, openai_embeddings)

        open_end_qa_ds = pd.DataFrame(json.load(open(f"data/test/{major}/{major}_closed_end.json", "r")))
        queries = open_end_qa_ds["question"].tolist()[:num_test]
        references = open_end_qa_ds["answer"].tolist()[:num_test]
        gold_docs = [[f"{item[0]['title']}\n{item[0]['text']}"] for item in open_end_qa_ds["paragraphs"].tolist()][:num_test]
        for query in queries:
            query_embedding = openai_embeddings.embed_query(query)
            results = vector_store.similarity_search_by_vector(query_embedding, k=retrieval_k)
            combined_text = " ".join([result.page_content for result in results])
            prompt = f"""
            Based on the following text, answer the question:\n\n{combined_text}\n\nQuestion: {query}.
            You should present a concise, definitive response, devoid of additional elaborations, and keep the same writing style as the found source text if the question is a open-ended question else you can answer just Yes or No.
            """
            response = llm(prompt)
            response = response.split("Answer: ")[-1].strip().replace(".", "")
            print(f"Query: {query} | Answer: {response} | Reference: {references[0]}")
            