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
python eval_qa.py --major MCS --kind-of-qa opened_end
python eval_qa.py --major DS --kind-of-qa opened_end
python eval_qa.py --major AM --kind-of-qa opened_end