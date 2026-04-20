# SemEval 2026 Task 12: Abductive Event Reasoning (AER)

An LLM-based agent system for **Abductive Event Reasoning** — given an observed real-world event and retrieved documents, the system infers the most plausible and direct cause through structured multi-step reasoning.

Built for the SemEval 2026 Task 12 shared task. Implements both classic and agentic reasoning strategies (Chain-of-Thought, Self-Consistency voting, Two-Pass verification, ReAct + Critic + Memory) with hybrid BM25 + semantic retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         run.py (CLI)                            │
│   Orchestrates experiments with ThreadPoolExecutor              │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
    ┌──────────▼──────────┐          ┌──────────▼──────────┐
    │   DataLoader        │          │   Evaluator         │
    │  docs.json +        │          │  Official metric:   │
    │  questions.jsonl    │          │  1.0 / 0.5 / 0.0   │
    └──────────┬──────────┘          └─────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────────┐
    │              Approach (reasoning strategy)           │
    │  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
    │  │  Baseline   │  │ SC-Refine    │  │ Two-Pass  │  │
    │  │  (CoT)      │  │ (7x voting)  │  │ (2-call)  │  │
    │  └─────────────┘  └──────────────┘  └───────────┘  │
    └──────────┬──────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐          ┌────────────────────┐
    │   DocumentRetriever │          │   ChatLLM          │
    │  BM25 + Semantic    │◄────────►│  OpenAI-compatible  │
    │  + RRF fusion       │          │  API with retry     │
    └─────────────────────┘          └────────────────────┘
```

## Project Structure

```text
llm-project/
├── data/               # Dataset files (SemEval 2026)
│   ├── sample/         #   Sample split (docs.json + questions.jsonl)
│   ├── train/          #   Training split
│   ├── dev/            #   Development split
│   └── test/           #   Test split
├── paper/              # Project report (ACL template)
├── src/
│   ├── approaches.py   # Reasoning strategies (Baseline, SC-Refine, TwoPass, etc.)
│   ├── dataloader.py   # Data loading and preprocessing
│   ├── evaluator.py    # Official evaluation metric implementation
│   ├── llm.py          # LLM API wrapper with exponential backoff retry
│   ├── prompts.py      # Prompt templates (CoT, Conservative, Evidence-Anchored, Balanced)
│   └── retriever.py    # Hybrid BM25 + semantic retrieval with RRF
├── tests/              # Unit tests
│   ├── test_parse_answer.py
│   ├── test_evaluator.py
│   └── test_post_processing.py
├── requirements.txt
├── .env.example        # Environment variable template
└── run.py              # Main entry point
```

## Quick Start

### 1. Install Dependencies

Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

The project supports any OpenAI-compatible API (DeepSeek, OpenAI, vLLM, etc.):

```
MODEL_NAME=deepseek-chat
API_KEY=your_api_key_here
BASE_URL=https://api.deepseek.com/v1
MAX_WORKERS=4
```

### 3. Run Experiments

```bash
python run.py
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## CLI Arguments

| Argument | Default | Description |
| :--- | :----: | :--- |
| `--approach` | `baseline` | Reasoning strategy: `baseline`, `sc_refine`, `conservative`, `lightweight_sc`, `twopass_real`, `agentic_react` |
| `--prompt_name` | `cot` | Prompt template: `cot`, `conservative`, `evidence_anchored`, `balanced` |
| `--top_k` | `10` | Number of documents to retrieve per query (0 = use all) |
| `--no_retrieval` | `False` | Disable retrieval and use the full document set as context |
| `--use_full_content` | `False` | Use full document text for retrieval instead of title+snippet |
| `--use_gpu` | `False` | Enable GPU acceleration for semantic retrieval |
| `--use_per_option` | `False` | Per-option weighted retrieval (event 2x + each option 1x) |
| `--docs_path` | `data/dev/docs.json` | Path to document corpus |
| `--questions_path` | `data/dev/questions.jsonl` | Path to questions file |
| `--submission_path` | `submission.jsonl` | Output submission file path |
| `--output_dir` | `results` | Directory to save detailed results JSON |

## Approach Overview

| Approach | API Calls | Strategy |
| :--- | :----: | :--- |
| `baseline` | 1 | Zero-shot Chain-of-Thought reasoning |
| `conservative` | 1 | Precision-focused, only select high-confidence options |
| `lightweight_sc` | 3 | Option-level voting with 3 samples |
| `sc_refine` | 7 | Self-Consistency with option-level voting + D-option penalty |
| `twopass_real` | 2 | Pass 1: liberal candidate selection, Pass 2: strict causal verification |
| `agentic_react` | 3-5 | Dynamic router + iterative ReAct retrieval + critic reflection + persistent memory |

## Recommended Command (Best Result)

```bash
python run.py \
  --approach sc_refine \
  --prompt_name balanced \
  --top_k 10 \
  --use_gpu \
  --use_full_content \
  --use_per_option
```

## Evaluation Metric

The official SemEval 2026 Task 12 metric:

| Condition | Score |
| :--- | :----: |
| Perfect match (P = G) | 1.0 |
| Partial match (P ⊂ G, no wrong selections) | 0.5 |
| Any wrong selection or empty answer | 0.0 |

**Key insight**: Wrong selection is catastrophic (0 points), while missing some correct answers gives partial credit (0.5 points). This asymmetry motivates our conservative strategies.

## License

Apache 2.0
