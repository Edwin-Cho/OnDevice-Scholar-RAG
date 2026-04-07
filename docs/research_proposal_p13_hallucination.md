# Research Proposal: Parametric Knowledge Suppression in On-Device RAG

> **Status:** Draft v0.1 — 2026-04-05  
> **System:** OnDevice Scholar RAG (Qwen2.5-3B-Instruct + FAISS + BGE-small-en-v1.5)  
> **Empirical Basis:** `reports/eval_basic_20260405_234102.json` (18 questions, warnings_per_query = 0.22)

---

## Title

**"Lightweight Numeric Hallucination Suppression in Sub-5B RAG Systems:  
A Study on Parametric vs. Contextual Knowledge Conflict"**

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are expected to ground their answers in retrieved documents. However, even with explicit prompt-level constraints (Rules 1–13), sub-5B language models (LLMs) operating on edge hardware continue to insert specific numerical figures—percentages, scores, counts—that are absent from the retrieved context. This phenomenon, which we term **P13 (Parametric Numeric Override)**, arises from a fundamental conflict between a model's parametric memory (weights-encoded pretraining knowledge) and its episodic/contextual memory (retrieved passages). This proposal outlines a **Tiered Suppression Pipeline** that requires no fine-tuning, is deployable on Apple Silicon (MPS), and is evaluated with a reproducible RAGAS-based benchmark we developed in-house.

---

## 1. Problem Statement

### 1.1 Empirical Observation

In our OnDevice Scholar RAG system, after applying:
- 13 explicit prompt rules (citation enforcement, verbatim numerics, comparison completeness)
- Comparison query sub-retrieval (P15)
- Metric label validation (P12 false-positive filtering)

the evaluation metric `warnings_per_query` converged to **0.22** (4 warnings / 18 queries). All 4 remaining warnings are P13-type: the LLM generates a specific percentage (e.g., `62.1%`) that does not appear verbatim in any of the 5 retrieved chunks.

### 1.2 Root Cause

LLMs store factual knowledge in their weights during pretraining. For well-covered domains (e.g., ML papers), the model "remembers" specific figures from training data and may prefer this parametric knowledge over retrieved context—even when instructed otherwise.

$$P_{\text{parametric}}(n) \gg P_{\text{contextual}}(n) \implies \text{hallucinated number } n \text{ is generated}$$

This is a known **knowledge conflict** problem (Xie et al., 2024) that prompt engineering cannot fully resolve.

---

## 2. Research Gap

| Approach | Key Reference | Limitation |
|---|---|---|
| Prompt constraints | — (ours) | Ceiling effect; LLM ignores instructions |
| Context-Aware Decoding (CAD) | Shi et al., 2023 | Validated on 7B+ models; edge latency unstudied |
| Contrastive Decoding | Li et al., 2023 | Requires two model forward passes; 2× latency |
| RAFT | Zhang et al., 2024 | Requires fine-tuning; high data preparation cost |
| Self-RAG | Asai et al., 2023 | Requires retraining with reflection tokens |
| Knowledge Conflicts Survey | Xie et al., 2024 | Classification only; no sub-5B empirical data |

**Core Research Gap:**  
*How much numeric hallucination can be suppressed in a sub-5B on-device RAG system without fine-tuning, and what is the latency-faithfulness Pareto frontier on Apple Silicon (MPS)?*

---

## 3. Proposed Method: Tiered Suppression Pipeline

### Architecture Overview

```
User Query
    │
    ▼
[Retrieval — P15 Sub-retrieval for comparison queries]
    │
    ▼
[Generation — Qwen2.5-3B-Instruct]
    │
    ├─── [Tier 1] Prompt Constraints (Rule 1–13)
    │         Always active. Reduces P13 rate from ~0.44 → 0.22 WPQ.
    │
    ├─── [Tier 2] Post-hoc Number Scrubbing         ← NEW
    │         P13 detector → replace hallucinated numbers in answer text
    │         Latency: O(1), no additional inference
    │         Target: WPQ → ~0.05
    │
    └─── [Tier 3] Lightweight CAD (opt-in)           ← RESEARCH
              Triggered only for P13-flagged queries
              2-pass: logit_ctx − α · logit_no_ctx
              Latency: ~1.5× per flagged query
              Target: faithfulness (chunk_overlap_ratio) +15pp
```

### Tier 2: Post-hoc Number Scrubbing

After generation, for each number flagged by P13:

```python
# Pseudocode
for num in p13_hallucinated_numbers(answer, retrieved_chunks):
    pattern = re.compile(rf'\b{re.escape(num)}%')
    replacement = f"[figure not reported in retrieved passages]"
    answer = pattern.sub(replacement, answer)
```

**Design rationale:** Preserve answer structure and citations; only replace the hallucinated figure with an honest epistemic hedge.

### Tier 3: Context-Aware Decoding (CAD)

Based on Shi et al. (2023), adapted for MPS inference:

$$\log P_{\text{CAD}}(y_t \mid y_{<t}, c, q) = \log P(y_t \mid y_{<t}, c, q) - \alpha \cdot \log P(y_t \mid y_{<t}, q)$$

Where:
- $c$ = retrieved context block
- $q$ = user query  
- $\alpha \in [0.1, 0.5]$ = suppression coefficient (hyperparameter)
- Pass 1: full prompt (context + query) → $\log P(\cdot \mid c, q)$
- Pass 2: query only → $\log P(\cdot \mid q)$ (parametric baseline)

**MPS optimization:** Batch both passes in a single `model.forward()` call with padded input to minimize overhead.

---

## 4. Experiments

| ID | Experiment | Metric | Baseline |
|---|---|---|---|
| **E1** | Current system (Rule 1–13 + P15) | WPQ, chunk_overlap | 0.22 WPQ, 0.678 overlap |
| **E2** | + Tier 2 scrubbing | WPQ, answer quality (human eval) | — |
| **E3** | + Tier 3 CAD, α ∈ {0.1, 0.2, 0.3, 0.5} | faithfulness, latency (s) | — |
| **E4** | Domain transfer (biomedical / legal) | WPQ on new eval_config.yaml | — |
| **E5** | RAFT baseline (fine-tuned Qwen2.5-3B) | faithfulness | — (comparison) |

### Evaluation Protocol

- **Quantitative:** `scripts/evaluate_rag.py` (BASIC mode + optional OpenAI RAGAS mode)
- **Qualitative:** Human evaluation of 5 scrubbed answers (accuracy of hedge insertion)
- **Latency:** `time.perf_counter()` around `generator.generate()` on MPS

---

## 5. Related Work

### 5.1 Knowledge Conflict in LLMs
> Xie, T. et al. (2024). *Knowledge Conflicts for LLMs: A Survey.* arXiv:2403.08319.

Provides the theoretical framework for parametric vs. contextual knowledge conflict. Classifies conflict types and intervention strategies. **Our work provides the first empirical P13 rate measurement for sub-5B RAG on edge hardware.**

### 5.2 Context-Aware Decoding
> Shi, W. et al. (2023). *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding.* arXiv:2305.14739.

Direct algorithmic foundation for Tier 3. Demonstrates logit subtraction reduces context-ignoring behavior. **Our contribution: edge hardware (MPS) deployment + opt-in triggering policy.**

### 5.3 Contrastive Decoding
> Li, X. et al. (2023). *Contrastive Decoding: Open-ended Text Generation as Optimization.* arXiv:2309.09117.

Motivates the expert-amateur logit gap idea. Extended by Shi et al. to RAG settings.

### 5.4 RAFT
> Zhang, T. et al. (2024). *RAFT: Adapting Language Model to Domain Specific RAG.* arXiv:2403.10131.

Fine-tuning baseline for comparison in E5. **Key contrast:** RAFT requires curated (question, oracle_doc, distractor_docs) triples; our pipeline requires none.

### 5.5 Self-RAG
> Asai, A. et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv:2310.11511.

Represents the retraining-based upper bound. Comparison motivates the value of our training-free approach.

### 5.6 FActScore
> Min, S. et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* arXiv:2305.14251.

Evaluation methodology reference. Our P13 metric is a domain-specific lightweight variant targeting numeric atomic facts.

---

## 6. Expected Contributions

1. **Empirical:** First systematic P13 numeric hallucination rate measurement in sub-5B on-device RAG (Qwen2.5-3B, Apple Silicon MPS)
2. **Algorithmic:** Tiered Suppression Pipeline — prompt → scrubbing → CAD — with clear latency-faithfulness tradeoffs
3. **Practical:** Training-free deployment; no GPU required; reproducible via open-source `scripts/evaluate_rag.py`
4. **Transferability:** Domain-agnostic framework demonstrated across ML, biomedical, legal domains (E4)
5. **Efficiency insight:** Characterization of α sensitivity for CAD in sub-5B models vs. 7B+ findings in literature

---

## 7. Implementation Roadmap

| Phase | Task | File |
|---|---|---|
| **P1 (immediate)** | Tier 2 scrubbing in `generate()` | `app/pipeline/generator.py` |
| **P2 (short-term)** | Tier 3 CAD 2-pass `generate()` | `app/pipeline/generator.py` |
| **P3 (mid-term)** | E3 α sweep + latency profiling | `scripts/evaluate_rag.py` |
| **P4 (long-term)** | E4 domain transfer (bioRxiv ingest) | `download_papers.sh` + `eval_config.yaml` |
| **P5 (optional)** | E5 RAFT baseline (QLoRA on Qwen2.5-3B) | separate `scripts/train_raft.py` |

---

## 8. Limitations & Future Work

- **P13 scrubbing** may over-hedge: valid numbers from LLM's parametric knowledge that happen to be correct will also be replaced
- **CAD** doubles inference time; practical for interactive use but may be too slow for batch processing
- Evaluation limited to 18 questions; needs expansion for statistical significance (n ≥ 100)
- All experiments on single hardware platform (Apple Silicon M-series); CUDA/CPU generalization needed

---

*Generated from: OnDevice Scholar RAG evaluation pipeline — `eval_basic_20260405_234102.json`*
