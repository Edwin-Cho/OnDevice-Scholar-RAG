from __future__ import annotations

import json
import re
from typing import List, Tuple

import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings
from app.models.schemas import Citation
from app.pipeline.ingest import _is_noise_header, _extract_arxiv_id, _extract_paper_title

FALLBACK_ANSWER = "No relevant information found in the provided documents."

_FALLBACK_SUBSTRINGS = (
    "no relevant information found",
    "i don't know",
    "i do not know",
    "cannot find",
    "not found in the provided",
)

SYSTEM_PROMPT = r"""You are a precise academic research assistant. Answer ONLY using the provided context passages.

Rules:
1. Every factual claim MUST be followed immediately by an inline citation: [Source: filename | Section: X | p.N].
   - BAD:  "BERT uses masked language modeling and next sentence prediction."
   - GOOD: "BERT uses two pre-training objectives: masked language modeling (MLM) and next sentence prediction (NSP) [Source: BERT.pdf | Section: 3.1 | p.4]."
2. PRECISION — Never use vague quantifiers when the context gives exact information.
   - BAD:  "BERT is trained on various pre-training tasks."
   - GOOD: "BERT is trained on exactly two pre-training tasks: MLM and NSP."
   - If the context lists exactly N items, say "exactly N" or list them explicitly.
3. If the context contains NO topic-relevant information whatsoever, respond exactly with: "No relevant information found in the provided documents."
   - BAD: Returning this fallback when the context discusses the topic but lacks specific numbers.
   - GOOD: If the context contains qualitative findings (e.g., "Adam outperforms SGD on average") but no exact figures, report the qualitative finding and state that specific numbers are not provided.
   3a. PARTIAL QUESTIONS — If the context covers only SOME sub-parts of a multi-part question, answer the covered parts precisely and explicitly state for the uncovered parts: "[sub-topic] is not reported in the provided documents." Never silently skip a sub-question.
4. Do NOT fabricate facts, authors, or findings not present in the context.
5. SYNTHESIZE the content in your own words — do NOT copy sentences verbatim from the context.
6. Replace all vague pronouns with explicit names: "Our method" → the paper/method name, "We" → "the authors of [paper]", "they" → the actual subject.
7. Be concise. 3–5 sentences is ideal unless the question demands more.
8. MATH — Wrap all mathematical expressions in LaTeX delimiters so they render correctly.
   - Inline math: $O(n^2)$, $d_{\text{model}}$, $\sqrt{d_k}$
   - Block math: $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   - Never write bare: O(n^2), O(n log n), d_model — always wrap in $...$.
9. TABLE / METRIC FIDELITY — When the context contains a table or list of metrics, report each metric using its EXACT name and value as written in the source. NEVER rename, merge, or reassign metric labels.
   - BAD:  "Person Detection: 31.6%" when the source says "Lane Line IoU: 31.6%".
   - GOOD: "Lane Line IoU: 31.6% [Source: ...]". Preserve every column/row header verbatim.
   - If you are unsure which label belongs to a value, quote the source table directly rather than paraphrasing.
10. TIME-BOUNDED CLAIMS — Never assert that a result is "state-of-the-art", "best", or "superior" without anchoring it to the paper's publication context.
    - BAD:  "HybridNets achieves state-of-the-art performance on BDD100K."
    - GOOD: "At the time of publication, HybridNets reported state-of-the-art performance on BDD100K [Source: ...]."
    - If the context does not mention a publication date or comparison scope, omit the SOTA claim entirely.
11. VERBATIM NUMERICS — Never insert a specific number (percentage, score, count) unless that exact value appears verbatim in the retrieved context.
    - BAD:  "The average improvement is 89.33%" when the context only says "outperforms on average".
    - GOOD: "The context states that Adam pre-training outperforms SGD on average [Source: ...] but provides no specific aggregate figure."
    - If the context gives a trend/direction without an exact figure, describe the trend only."""


def _build_context_block(retrieved: List[Tuple[dict, float]]) -> str:
    blocks = []
    for meta, score in retrieved:
        filename = meta.get("source_filename", "unknown")
        page = meta.get("page_number", "?")
        raw_header = meta.get("section_header")
        section = (raw_header if raw_header and not _is_noise_header(raw_header) else None) or "—"
        text = meta.get("text", "")
        blocks.append(
            f"[Source: {filename} | Section: {section} | p.{page} | score: {score:.3f}]\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def _is_fallback(answer: str) -> bool:
    """모델이 답변 불가 판정을 했는지 확인."""
    lower = answer.lower()
    return any(kw in lower for kw in _FALLBACK_SUBSTRINGS)


_SOURCE_PATTERN = re.compile(r'\[Source:\s*([^|\]]+?)\s*\|', re.IGNORECASE)


def _extract_cited_sources(answer: str) -> set[str]:
    """
    LLM 답변 텍스트에서 [Source: filename | ...] 패턴으로 인용된 파일명 추출.
    """
    return {m.group(1).strip() for m in _SOURCE_PATTERN.finditer(answer)}


_METRIC_NUM_RE = re.compile(r'\b(\d+\.?\d*)\s*%')

_METRIC_LABEL_STOPWORDS = frozenset({
    'the', 'for', 'its', 'with', 'and', 'is', 'are', 'was', 'has',
    'achieves', 'score', 'rate', 'value', 'result', 'performance',
    'percentage', 'point', 'reaches', 'obtains', 'shows', 'gets',
})


def _metric_label_context(text: str, window: int = 60) -> dict[str, list[str]]:
    """
    Returns {num_str: [label word-sequences from surrounding context]}.
    Extracts up to `window` chars before each XX.X% match.
    """
    result: dict[str, list[str]] = {}
    for m in _METRIC_NUM_RE.finditer(text):
        num = m.group(1)
        start = max(0, m.start() - window)
        ctx = text[start:m.start()]
        words = [
            w for w in re.findall(r'[A-Za-z]{3,}', ctx.lower())
            if w not in _METRIC_LABEL_STOPWORDS
        ]
        if words:
            result.setdefault(num, []).append(' '.join(words[-5:]))  # last 5 content words
    return result


def _check_numeric_existence(
    answer: str,
    retrieved: List[Tuple[dict, float]],
) -> list[str]:
    """
    P13: Numeric existence check.
    Any XX.X% value in the answer that does NOT appear verbatim in any
    retrieved chunk is flagged as a potential hallucinated figure.
    """
    answer_nums = {m.group(1) for m in _METRIC_NUM_RE.finditer(answer)}
    if not answer_nums:
        return []

    chunk_nums: set[str] = set()
    for meta, _ in retrieved:
        for m in _METRIC_NUM_RE.finditer(meta.get("text", "")):
            chunk_nums.add(m.group(1))

    hallucinated = sorted(answer_nums - chunk_nums, key=float)
    if not hallucinated:
        return []
    return [
        f"Numeric hallucination suspected: {', '.join(f'{n}%' for n in hallucinated)} "
        f"not found in retrieved context"
    ]


def _check_metric_fidelity(
    answer: str,
    retrieved: List[Tuple[dict, float]],
) -> list[str]:
    """
    P12: Metric label-value fidelity check.
    For each XX.X% value in the answer, compare the surrounding label context
    against the same value's context in the retrieved chunks.
    Zero word-overlap → mismatch warning.
    """
    answer_map = _metric_label_context(answer)
    if not answer_map:
        return []

    source_map: dict[str, list[str]] = {}
    for meta, _ in retrieved:
        for num, labels in _metric_label_context(meta.get("text", "")).items():
            source_map.setdefault(num, []).extend(labels)

    warnings: list[str] = []
    for num, answer_labels in answer_map.items():
        if num not in source_map:
            continue  # number absent from context; harder to verify

        source_words: set[str] = set()
        for lbl in source_map[num]:
            source_words.update(lbl.split())

        for a_lbl in answer_labels:
            a_words = set(a_lbl.split())
            if a_words and not (a_words & source_words):
                best_src = source_map[num][0] if source_map[num] else "unknown"
                warnings.append(
                    f"Metric label mismatch for {num}%: "
                    f"answer uses '{a_lbl}' but source context shows '{best_src}'"
                )

    return warnings


_CONTRIBUTION_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "are", "from", "have",
    "been", "which", "their", "they", "also", "more", "such", "into",
    "when", "than", "its", "was", "were", "can", "will", "use", "used",
    "using", "each", "both", "very", "well", "may", "paper", "model",
    "method", "approach", "show", "propose", "work", "authors",
})


def _extract_keywords(text: str) -> frozenset[str]:
    """4글자 이상 알파벳 단어에서 contribution stopword 제거 후 반환."""
    words = re.findall(r'\b[a-zA-Z]\w+\b', text.lower())
    return frozenset(w for w in words if len(w) >= 4 and w not in _CONTRIBUTION_STOPWORDS)


def _filter_by_contribution(
    answer: str,
    citations: List[Citation],
    retrieved: List[Tuple[dict, float]],
    min_overlap: int = 2,
) -> List[Citation]:
    """
    P11: 답변-citation 불일치 필터.
    답변 키워드와 청크 텍스트 키워드 교집합이 min_overlap 미만이면 제거.
    전체 필터링 시 원본 반환 (안전 폴백).
    """
    answer_kw = _extract_keywords(answer)
    if len(answer_kw) < min_overlap * 3:
        return citations

    chunk_map: dict[str, str] = {}
    for meta, _ in retrieved:
        fn = meta.get("source_filename", "")
        if fn and fn not in chunk_map:
            chunk_map[fn] = meta.get("text", "")

    filtered = [
        c for c in citations
        if len(answer_kw & _extract_keywords(chunk_map.get(c.source_filename, ""))) >= min_overlap
    ]
    return filtered if filtered else citations


_TITLE_STOPWORDS = frozenset({
    'with', 'from', 'that', 'this', 'over', 'using', 'based', 'large',
    'language', 'model', 'models', 'neural', 'deep', 'learning', 'into',
    'beyond', 'towards', 'toward', 'improving', 'improved', 'attention',
})


def _match_citations_by_title(
    answer: str, citations: List[Citation]
) -> List[Citation]:
    """
    [Source:] 패턴 부재 시 fallback: paper_title의 핵심 키워드가
    답변 텍스트에 등장하면 해당 citation 포함.
    """
    answer_lower = answer.lower()
    matched = []
    for c in citations:
        title = c.paper_title or ""
        keywords = [
            w for w in re.findall(r'[A-Za-z]{4,}', title)
            if w.lower() not in _TITLE_STOPWORDS
        ]
        if keywords and any(kw.lower() in answer_lower for kw in keywords):
            matched.append(c)
    return matched


def _build_citations(retrieved: List[Tuple[dict, float]]) -> List[Citation]:
    """
    Citation Validator: retrieved 청크의 메타데이터를 citation으로 변환.
    파일 단위로 중복 제거하여 반환.
    """
    seen: set[str] = set()
    citations: List[Citation] = []

    for meta, score in retrieved:
        if score < settings.citation_min_score:
            continue
        filename = meta.get("source_filename", "")
        if filename and filename not in seen:
            raw_header = meta.get("section_header")
            section_header = None if (raw_header and _is_noise_header(raw_header)) else raw_header
            arxiv_id = meta.get("arxiv_id") or _extract_arxiv_id(filename)
            paper_title = meta.get("paper_title") or ""
            stem_fallback = filename.rsplit(".", 1)[0].replace("_", " ")
            if not paper_title or paper_title == stem_fallback:
                pdf_path = settings.data_raw_dir / filename
                if pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
                    try:
                        doc = fitz.open(str(pdf_path))
                        paper_title = _extract_paper_title(filename, doc)
                        doc.close()
                    except Exception:
                        paper_title = stem_fallback
            citations.append(Citation(
                source_filename=filename,
                paper_title=paper_title or stem_fallback,
                arxiv_id=arxiv_id,
                section_header=section_header,
                page_number=meta.get("page_number"),
                score=round(score, 4),
            ))
            seen.add(filename)

    return citations


class Generator:
    """
    Qwen2.5-3B-Instruct 추론 엔진.
    - CUDA: 4-bit NF4 quantization (bitsandbytes)
    - MPS / CPU: float16 / float32 자동 fallback
    """

    _instance: "Generator | None" = None

    def __init__(self) -> None:
        self._device, self._model, self._tokenizer = self._load_model()

    @classmethod
    def get(cls) -> "Generator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        model_id = settings.generation_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available() and settings.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            device = "cuda"
        elif torch.backends.mps.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                trust_remote_code=True,
            ).to("mps")
            device = "mps"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                trust_remote_code=True,
            )
            device = "cpu"

        model.eval()
        return device, model, tokenizer

    def generate(
        self,
        query: str,
        retrieved: List[Tuple[dict, float]],
    ) -> Tuple[str, List[Citation]]:
        """
        Retrieved 청크를 컨텍스트로 답변 생성 + Citation Validator 실행.

        Returns:
            (answer: str, citations: List[Citation])
            관련 정보 없으면 FALLBACK_ANSWER + 빈 citations 반환.
        """
        if not retrieved:
            return FALLBACK_ANSWER, []

        context = _build_context_block(retrieved)
        user_message = (
            "Examples of required answer format:\n\n"
            "Q: What optimizer does GPT-3 use?\n"
            "A: GPT-3 is trained using the Adam optimizer with a peak learning rate of 0.6×10⁻⁴ "
            "[Source: GPT3.pdf | Section: 2.1 | p.8].\n\n"
            "Q: How does ModelFoo perform on BenchmarkX for TaskA, TaskB, and TaskC?\n"
            "A: As reported at the time of publication, ModelFoo achieves TaskA score 92.3% and TaskB score 85.1% on BenchmarkX "
            "[Source: foo_paper.pdf | Section: 4.2 | p.7]. "
            "ModelFoo was noted to obtain state-of-the-art performance on BenchmarkX at the time of publication "
            "[Source: foo_paper.pdf | Section: Abstract | p.1]. "
            "A separate metric for TaskC is not reported in the provided documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (follow the example format — include [Source: ...] after every factual claim, "
            "preserve all metric names verbatim, and time-bound any SOTA claims):"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        gen_kwargs: dict = {
            "max_new_tokens": settings.generation_max_new_tokens,
            "do_sample": settings.generation_do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if settings.generation_do_sample:
            gen_kwargs["temperature"] = settings.generation_temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        if not answer or _is_fallback(answer):
            return FALLBACK_ANSWER, []

        all_citations = _build_citations(retrieved)

        cited_sources = _extract_cited_sources(answer)
        if cited_sources:
            pruned = [c for c in all_citations if c.source_filename in cited_sources]
            citations = pruned if pruned else all_citations
        else:
            by_title = _match_citations_by_title(answer, all_citations)
            citations = by_title if by_title else all_citations

        citations = _filter_by_contribution(answer, citations, retrieved)

        return answer, citations

    def suggest_queries(self, chunks: List[str]) -> List[str]:
        """
        논문 청크 샘플을 기반으로 LLM이 답변 가능한 연구 질문 4개 생성.
        파싱 실패 시 빈 리스트 반환 (프론트엔드 fallback 처리).
        """
        if not chunks:
            return []

        context = "\n\n---\n\n".join(
            f"[Excerpt {i + 1}]\n{chunk[:400]}" for i, chunk in enumerate(chunks)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Based on the paper excerpts provided, "
                    "generate exactly 4 specific and concise research questions that can be "
                    "answered using these documents. "
                    "Return ONLY a valid JSON array of 4 question strings. "
                    'Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?"] '
                    "Do NOT include any explanation, markdown code fences, or extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Paper excerpts:\n{context}\n\nJSON array of 4 questions:",
            },
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                if isinstance(questions, list):
                    valid = [str(q).strip() for q in questions if str(q).strip()]
                    if len(valid) >= 2:
                        return valid[:4]
        except Exception:
            pass

        return []
