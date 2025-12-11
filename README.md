# Course Information Question-Answering System for University Students: Graph-Integrated RAG (HippoRAG-2)

## Introduction
This undergraduate thesis project focuses on designing and developing a specialized **Course Information Question-Answering (QA) System** for university students. The system is built upon the **Retrieval-Augmented Generation (RAG)** framework, with the primary goal of supporting fast, accurate, and efficient course lookup. The research was conducted by the team with 2 students of the Data Science program, Faculty of Mathematics and Computer Science, at the University of Science, Vietnam National University, Ho Chi Minh City (VNU-HCM). Besides, all research information in `report.pdf` file are written in Vietnamese.

The project addresses the limitations of manual course lookup and traditional support channels, which often suffer from response delays and inconsistent information, by proposing an automated QA system. The system aims to leverage course knowledge effectively while generating natural, easy-to-understand answers appropriate to the student's query context. 

## Goals and Scope
The core objective of the research is to evaluate and compare two advisory systems: the **baseline RAG model (Simple RAG)** and a **Graph-integrated RAG model (HippoRAG-2 / Graph-based RAG)**. This comparison is conducted through quantitative metrics and qualitative assessment based on pre-defined real-world scenarios.

The system focuses specifically on retrieving and responding to course information in an automated QA format, utilizing structured input data. The project defined **10 specific study scenarios**, including:
*   Retrieving detailed information about a single module (e.g., prerequisites, semester offered, or learning outcomes).
*   Retrieving and summarizing learning outcomes for a group of courses (up to five).
*   Retrieving prerequisite relationships among a group of modules.
*   Retrieving instructor names for a group of courses.

The scope is intentionally limited to information retrieval and response, excluding functions like personalized course recommendations based on student history or career goals.

## Architecture and Methodology

The system adheres to the standard RAG architecture, comprising three main components: data preprocessing and course knowledge construction, relevant document retrieval based on queries, and automatic answer generation using a Large Language Model (LLM).

### 1. Data Processing and Input

Input data was extracted from the module handbooks (in `.docx` format) of three academic programs at the Faculty of Mathematics and Computer Science, VNU-HCM: **Data Science (DS)**, **Applied Mathematics (AM)**, and **Mathematics and Computer Science (MCS)**. The specifics are presented within the `report.pdf`.

**Preprocessing Pipeline:**
A complete and automated data processing pipeline was developed to clean, normalize, and restructure the raw `.docx` data into a consistent JSON format. A key challenge addressed was the inconsistency and errors in prerequisite course names, particularly permutation errors.

*   **Prerequisite Standardization:** The `PrerequisiteExtractor` module was deployed to match and standardize incorrectly formatted prerequisite names. Comparative evaluation showed that the **Jaccard coefficient** provided superior performance (average Accuracy of 0.96 and Macro F1-Score of 0.95) for handling permutation errors compared to other methods like Levenshtein or Longest Common Subsequence (LCS).

**Output Data:** The output is standardized course data in JSON format, used to generate the LLM response, which is primarily prioritized in English.

```
{
    "MTH00010": [
        "course id: MTH00010",
        "course name: Analysis 1A",
        "semester: odd",
        "teacher name: Ong Thanh Hai",
        "course type: Compulsory",
        "required prerequisites: none",
        "learning outcomes: The objective of the module is to equip 
        students with the basic knowledge of the foundation of calculus
        as the foundation for specialized modules.",
        "content: The course covers the basics of real numbers, sequences
        and series of real numbers."
    ]
}
```

### 2. RAG Model Comparison
The thesis compares Simple RAG against the extended architecture, **HippoRAG-2**.

#### HippoRAG-2 (Graph-based RAG)
HippoRAG-2, a hybrid GraphRAG model, integrates a knowledge graph to exploit the complex relationships between courses, such as prerequisites or related knowledge.

*   **Offline Indexing:** An LLM (acting as an "artificial neocortex") is used to extract entities and triples (subject-predicate-object) from course passages. This information is used to build a **Knowledge Graph** containing Document Nodes (red) and Entity Nodes (blue). Edges are created for semantic relations and synonymous entities (using L2-normalized cosine similarity with a threshold of 0.8).
*   **Online Retrieval:** The system extracts query entities and performs Dense Passage Retrieval (DPR). Relevant nodes are identified as **Seed Nodes**. The **Personalized PageRank (PPR)** algorithm is then applied to traverse the graph and calculate relevance scores for all nodes, prioritizing those related to the query's context. The top 5 highest-scoring document nodes are retrieved as context for the final generation stage.

#### Simple RAG
The Simple RAG model serves as the baseline, maintaining the core RAG architecture but omitting the entity extraction, triple generation, and knowledge graph construction phases. Retrieval relies solely on comparing the vector representation of the query against the vector representations of the course passages.

## Models and Technology Used
| Component | Models Used (Examples) | Role |
| :--- | :--- | :--- |
| **LLMs (Generation/Extraction)** | **GPT-4o mini**, **Llama-3.1-8B**. | Entity/Triple extraction (offline) and final answer generation (online). |
| **Embedding Models (Retrieval/Indexing)** | **NV-Embed-v2**, **GritLM-7B**, text-embedding-3-small, Contriever. | Creating vector representations of passages/entities and performing dense retrieval (DPR). |
| **Sparse Retrieval** | BM25. | Used as a traditional baseline method for retrieval comparison. |
| **Algorithms** | Personalized PageRank (PPR), Jaccard Coefficient. | Ranking nodes in the KG for Graph-based RAG; string matching during data preprocessing. |

## Key Findings and Evaluation
The models were evaluated using three test sets: `closed_end` (simple, short answers), `opened_end` (simple, natural language answers), and `multihop` (complex reasoning, integrating information from multiple courses),.

### Performance with State-of-the-Art Embeddings (GPT-4o mini)
When using modern, powerful embedding models like **NV-Embed-v2** and **GritLM-7B** for retrieval:
*   **Retrieval:** Simple RAG demonstrated very high and competitive retrieval performance across all datasets (e.g., DS average R@1 reached 71.4%, MCS average R@1 reached 71.1%),. The performance of Simple RAG was superior to traditional methods like BM25 (e.g., AM R@1=35.8).
*   **Graph Impact:** The integration of the graph structure in HippoRAG-2 provided only a small, non-significant performance boost when these strong embedding models were already used,.
*   **Generation Quality:** The difference between Simple RAG and Graph-based RAG in terms of language generation quality (measured by BLEU, METEOR, ROUGE-L, and AI-based evaluation by GPT-4) was generally **not clearly noticeable**.

### Performance with other models (Llama-3.1-8B + Contriever)
When evaluating the performance using the smaller LLM (Llama-3.1-8B) paired with the Contriever embedding model:
*   **Graph Superiority:** **Graph-based RAG showed clear superiority** over Simple RAG. The average **Recall@5** for Graph-based RAG across all three programs consistently exceeded **90%**, demonstrating that the knowledge graph successfully enhances retrieval performance and contextual relevance when weaker embedding models are utilized.

### Conclusion
The overall analysis suggests that **Simple RAG**, when equipped with high-quality, modern embedding models, presents a more suitable solution for this application. This approach achieves highly competitive performance while benefiting from a simpler architecture, faster processing time, and lower deployment cost compared to Graph-based RAG,.

## Future Work
Potential directions for future research include:
1.  **Data and Language Expansion:** Extending the system beyond course descriptions to include detailed syllabi and learning materials, and supporting the Vietnamese language.
2.  **Practical Deployment and Personalization:** Implementing the system as a chatbot integrated into student portals and expanding functionality to include course recommendations personalized by individual student learning history and career goals
---
## Repository Link
The project is built upon the HippoRAG repository:
<https://github.com/OSU-NLP-Group/HippoRAG/tree/main>
