# SemEval2026-AER
Course project for "Large Language Models for Software Engineering 25/26" @ PoliTo. Exploring SemEval 2026 Task 12: Abductive Event Reasoning (AER) to investigate real-world causal inference capabilities in LLM.

## Project Structure

```text
AER-Project/
│
├── data/               # Dataset files (SemEval 2026)
├── paper/              # Project report with ACL template
├── src/
│   ├── approaches.py   # Reasoning logic (Baseline, SC_refine, etc.)
│   ├── dataloader.py   # Data preprocessing and loading
│   ├── evaluator.py    # Evaluate model performance
│   ├── prompts.py      # different type of prompts(Balanced, Conservative, etc.)
│   ├── llm.py          # LLM API Wrapper (DeepSeek)
│   └── retriever.py    # Retrieves and ranks documents
├── requirements.txt    # python dependencies
└── run.py              # Main entry point for experiments
```

## Quick Start
1. Install dependencies
Clone the repository and install the required dependencies (Python 3.9+ recommended).
```bash
> pip install -r requirements.txt
```
2. Configuration
Create a `.env` file in the root directory to configure your LLM provider. The project supports any OpenAI-compatible API (e.g., DeepSeek, OpenAI, vLLM).
```
MODEL_NAME='model_name_here'
API_KEY='your_api_key_here'
BASE_URL='vendor_url_here'
MAX_WORKERS=<positive integer>
```
3. Running Experiments
Run the main script to start the evaluation pipeline.
```bash
> python run.py
```
#### CLI Usage & Arguments
You can customize the experiment using command-line arguments:
|Argument|Default|Description|
| :--- | :----: | :--- |
|`--approach`|`baseline`|Reasoning strategy. Options: `baseline`, `sc_refine`, `conservative`, `twopass_real`|
|`--prompt_name`|`cot`|Prompt template. Options: `cot`, `conservative`, `evidence_anchored`, `balanced`|
|`--top_k`|`10`|Number of documents to retrieve per query (0 = use all docs).|
|`--no_retrieval`|`False`|Disable retrieval and use the full document set context.|
|`--use_full_content`|`False`|Use full document text for retrieval instead of title+snippet.|
|`--use_gpu`|`False`|Enable GPU acceleration for semantic retrieval (Sentence-Transformers).|
|`--use_per_option`|`False`|Use per-option retrieval (retrieve for event + each option).|

In order to reproduce our SOTA result, recommended command is:
```bash
> python run.py \
--approach sc_refine \
--prompt_name balanced \
--top_k 10 \
--use_gpu \
--use_full_content \
--use_per_option
```


