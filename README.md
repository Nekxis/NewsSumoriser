# NewsSumoriser

#### NewsSummariser is a LangChain project that summarizes news on any topic, delivering concise reports of the most important events with the help of a chain of ai researching agents.

- Install and launch Ollama: `ollama serve`
- Create new environment: `conda env create -f environment.yml`
- Activate the new environment: `conda activate NewsSumoriser`
- Pull the model you intend to use: `ollama pull mixtral:8x7b-instruct-v0.1-q5_K_M`
- Run: `python3 main.py`

#### Other notes

The default model is hardcoded into the `lookup.py` file.
