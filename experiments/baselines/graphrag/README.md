# Graph-RAG baseline

A retrieval-augmented baseline: passages are retrieved from the TITAN graph
for a question, then an LLM answers in free text from that context — no
multi-hop path, no DSL, no SPARQL. Unlike `experiments/baselines/rag_baseline.py`
(retrieval + generation in one process), this baseline splits the two stages
so retrieval runs once and many model configurations can be generated against
the same cached contexts.

## Status

The two scripts every run script here depends on are **not yet included**:

- `graphrag_retrieve.py` — retrieves top-k graph passages per question into a
  contexts CSV (`--test`, `--top-k`, `--max-context-chars`, `--out`).
- `graphrag_generate.py` — generates an answer per question from a contexts
  CSV with a given model (`--contexts`, `--model`, `--quantization`,
  `--tp`/`--pp`, `--style` [`cot`/`nocot`], `--fewshot`, `--max-new`,
  `--max-model-len`, `--gpu-mem-util`, `--out`).

Drop both scripts into this directory once available; the run scripts below
are already wired to call them with the right flags and paths.

## Run scripts

All scripts are invoked from the repository root and read/write inside
`experiments/baselines/graphrag/`. `PYTHON_BIN` can be set to point at a
specific interpreter (defaults to `python` on `PATH`).

| Script | Test set | Models | GPUs |
|---|---|---|---|
| `run_graphrag_queue.sh` | 1,000-question smoke subset | Qwen2.5-7B | 1 |
| `run_graphrag_big.sh` | 1,000-question smoke subset | Qwen2.5-72B, Llama-3.3-70B | 2 (pipeline-parallel) |
| `run_graphrag_gptoss.sh` | 1,000-question smoke subset | gpt-oss-120b (CoT) | 2 (tensor-parallel) |
| `run_graphrag_budget.sh` | 1,500-row paired subset, k=30 retrieval | Qwen2.5-72B (NoCoT) | 2 |
| `run_graphrag_fair.sh` | 800-row matched subset | any (spec'd via CLI args) | per invocation |
| `run_graphrag_queue4.sh` | 800-row matched subset | Qwen2.5-72B, gpt-oss-120b | sequential, waits for free GPUs |
| `run_graphrag_full_A.sh` | full 6,772-question split (pair A) | Llama-3.3-70B | 2 |
| `run_graphrag_full_B.sh` | full 6,772-question split (pair B) | Qwen2.5-7B, gpt-oss-120b, Qwen2.5-72B | 2 |

`run_graphrag_budget.sh` is a paired ablation: it re-retrieves the same
1,500 gold-executable rows used elsewhere with a 3x larger passage budget
(k=30 vs k=10), isolating how much of the graph the reader is shown.

`run_graphrag_full_A.sh` and `run_graphrag_full_B.sh` are meant to run
concurrently on disjoint GPU pairs.
