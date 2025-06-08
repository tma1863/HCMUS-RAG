if [ -f .env ]; then
    # Load environment variables from the .env file
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
fi
# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
# python eval_qa.py --major MCS --kind-of-qa closed_end
# python eval_qa.py --major DS --kind-of-qa closed_end
# python eval_qa.py --major AM --kind-of-qa closed_end
# python eval_qa.py --major MCS --kind-of-qa opened_end
# python eval_qa.py --major DS --kind-of-qa opened_end
# python eval_qa.py --major AM --kind-of-qa opened_end
# python eval_qa.py --major DS --kind-of-qa multihop2
# python eval_qa.py --major DS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa_standard_rag.py --major DS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa_standard_rag.py --major MCS --kind-of-qa opened_end --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major MCS --kind-of-qa closed_end --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major AM --kind-of-qa opened_end --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major AM --kind-of-qa closed_end --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major DS --kind-of-qa opened_end --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major DS --kind-of-qa closed_end --embedding-model-name text-embedding-3-small

# python eval_qa.py --major DS --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major DS --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
# python eval_qa.py --major DS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa_standard_rag.py --major DS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa.py --major DS --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
# python eval_qa_standard_rag.py --major DS --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
# python eval_qa.py --major DS --kind-of-qa multihop2 --embedding-model-name facebook/contriever
# python eval_qa_standard_rag.py --major DS --kind-of-qa multihop2 --embedding-model-name facebook/contriever

# python eval_qa.py --major MCS --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
# python eval_qa_standard_rag.py --major MCS --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
# python eval_qa.py --major MCS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa_standard_rag.py --major MCS --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
# python eval_qa.py --major MCS --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
# python eval_qa_standard_rag.py --major MCS --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
# python eval_qa.py --major MCS --kind-of-qa multihop2 --embedding-model-name facebook/contriever
# python eval_qa_standard_rag.py --major MCS --kind-of-qa multihop2 --embedding-model-name facebook/contriever

python eval_qa.py --major AM --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
python eval_qa_standard_rag.py --major AM --kind-of-qa multihop2 --embedding-model-name text-embedding-3-small
python eval_qa.py --major AM --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
python eval_qa_standard_rag.py --major AM --kind-of-qa multihop2 --embedding-model-name GritLM/GritLM-7B
python eval_qa.py --major AM --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
python eval_qa_standard_rag.py --major AM --kind-of-qa multihop2 --embedding-model-name nvidia/NV-Embed-v2
python eval_qa.py --major AM --kind-of-qa multihop2 --embedding-model-name facebook/contriever
python eval_qa_standard_rag.py --major AM --kind-of-qa multihop2 --embedding-model-name facebook/contriever