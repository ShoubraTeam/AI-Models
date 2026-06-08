from schemas.language_clarity.language_clarity_eval_schema import LanguageClarityEvalSchema
from schemas import FinalSubagentResult
import re

LANGUAGE_CLARITY_THRESHOLD = 0.5  # normalized 0.0–1.0

# Thresholds for text metrics
MIN_WORD_COUNT           = 50
MAX_GRAMMAR_ERRORS_MINOR = 3   # <= 3 errors → minor issues, half penalty

def calc_text_metrics(proposal_text: str) -> dict:
    """
    Calculate text metrics using pure code — no LLM needed.

    Metrics:
        - word_count           : total number of words
        - avg_sentence_length  : average words per sentence
        - length_score         : 1.0 if word_count >= MIN_WORD_COUNT, else 0.0
        - grammar_error_count  : number of grammar/spelling errors detected
        - grammar_score        : 1.0 if 0 errors | 0.5 if <= 3 errors | 0.0 if > 3

    Args:
        proposal_text: The freelancer's proposal text.

    Returns:
        dict with all text metrics.
    """
    words     = proposal_text.split()
    sentences = [
        s.strip()
        for s in proposal_text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    word_count          = len(words)
    avg_sentence_length = round(word_count / len(sentences), 2) if sentences else 0.0
    length_score        = 1.0 if word_count >= MIN_WORD_COUNT else 0.0

    grammar_error_count = estimate_grammar_error_count(proposal_text)

    if grammar_error_count == 0:
        grammar_score = 1.0
    elif grammar_error_count <= MAX_GRAMMAR_ERRORS_MINOR:
        grammar_score = 0.5
    else:
        grammar_score = 0.0

    return {
        "word_count"          : word_count,
        "avg_sentence_length" : avg_sentence_length,
        "length_score"        : length_score,
        "grammar_error_count" : grammar_error_count,
        "grammar_score"       : grammar_score,
    }


def estimate_grammar_error_count(proposal_text: str) -> int:
    """
    Lightweight grammar heuristic used during local and parallel tests.

    language_tool_python starts a Java-backed LanguageTool server and can block at
    import/runtime when the local dependency is unavailable. This heuristic keeps
    the language clarity pipeline deterministic and non-blocking.
    """
    errors = 0
    stripped_text = proposal_text.strip()

    if not stripped_text:
        return 1

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", stripped_text)
        if sentence.strip()
    ]

    for sentence in sentences:
        first_alpha = next((char for char in sentence if char.isalpha()), "")
        if first_alpha and not first_alpha.isupper():
            errors += 1

    if re.search(r"\bi\b", stripped_text):
        errors += 1

    repeated_words = re.findall(
        r"\b([A-Za-z]+)\s+\1\b",
        stripped_text,
        flags=re.IGNORECASE,
    )
    errors += len(repeated_words)

    return errors


def calc_language_clarity_score(
    llm_eval    : LanguageClarityEvalSchema,
    text_metrics: dict
) -> float:
    """
    Compute the final language clarity score (normalized 0.0–1.0).

    Weights:
        is_clear                    → 3 points
        is_professional             → 3 points
        not has_misleading_phrasing → 2 points
        grammar_score               → 1 point  (0.0 / 0.5 / 1.0)
        length_score                → 1 point  (0.0 / 1.0)
        ────────────────────────────────────────
        Total                       → 10 points → divided by 10 → 0.0–1.0

    Returns:
        float score between 0.0 and 1.0
    """
    score = 0.0

    if llm_eval.is_clear:
        score += 3.0
    if llm_eval.is_professional:
        score += 3.0
    if not llm_eval.has_misleading_phrasing:
        score += 2.0

    score += text_metrics["grammar_score"] * 1.0
    score += text_metrics["length_score"]  * 1.0

    return round(score / 10.0, 2)  # normalize to 0.0–1.0


def build_reasons(
    llm_eval    : LanguageClarityEvalSchema,
    text_metrics: dict,
    accepted    : bool
) -> list[str]:
    """
    Build a list of specific, actionable reasons from flags and metrics.
    Returns acceptance reasons if accepted, rejection reasons if not.
    Each reason maps directly to one check — consumed by the SuperAgent.

    Args:
        llm_eval    : Output of LanguageClarityEvaluator.
        text_metrics: Output of calc_text_metrics.
        accepted    : Whether the proposal passed the threshold.

    Returns:
        List of reason strings (10–100 chars each to satisfy FinalSubagentResult).
    """
    reasons = []

    if accepted:
        if llm_eval.is_clear:
            reasons.append("The proposal is clear and easy to follow.")
        if llm_eval.is_professional:
            reasons.append("The tone is professional and appropriate for a client.")
        if not llm_eval.has_misleading_phrasing:
            reasons.append("No vague or misleading statements were found.")
        if text_metrics["grammar_score"] == 1.0:
            reasons.append("No grammar or spelling errors were detected.")
        if text_metrics["length_score"] == 1.0:
            reasons.append("The proposal length is sufficient and detailed.")
    else:
        if not llm_eval.is_clear:
            reasons.append("Proposal is hard to follow. Use clearer sentences.")
        if not llm_eval.is_professional:
            reasons.append("Tone doesn't sound professional to the client.")
        if llm_eval.has_misleading_phrasing:
            reasons.append("Proposal has vague claims. Back them up with specifics.")
        if text_metrics["grammar_error_count"] > MAX_GRAMMAR_ERRORS_MINOR:
            reasons.append(
                f"Found {text_metrics['grammar_error_count']} grammar errors. Proofread first."
            )
        if text_metrics["word_count"] < MIN_WORD_COUNT:
            reasons.append(
                f"Proposal is too short ({text_metrics['word_count']} words). Add more detail."
            )

    return reasons


def calc_language_clarity_result(
    llm_eval     : LanguageClarityEvalSchema,
    proposal_text: str,
    threshold    : float = LANGUAGE_CLARITY_THRESHOLD
) -> FinalSubagentResult:
    """
    Full processing pipeline:
        1. Compute text metrics via pure code (word count, sentence length, grammar)
        2. Compute final normalized score (0.0–1.0)
        3. Apply rule-based acceptance decision
        4. Build specific reasons from flags
        5. Return FinalSubagentResult

    Args:
        llm_eval      : Output of LanguageClarityEvaluator.
        proposal_text : The freelancer's proposal (needed for text metrics).
        threshold     : Minimum passing score (default: 0.5).

    Returns:
        FinalSubagentResult consumed by the SuperAgent.
    """
    # Step 1 — text metrics (pure code, no LLM)
    text_metrics = calc_text_metrics(proposal_text=proposal_text)

    # Step 2 — final normalized score
    score = calc_language_clarity_score(
        llm_eval     = llm_eval,
        text_metrics = text_metrics
    )

    # Step 3 — rule-based acceptance decision
    accepted = score >= threshold

    # Step 4 — specific reasons from flags
    reasons = build_reasons(
        llm_eval     = llm_eval,
        text_metrics = text_metrics,
        accepted     = accepted
    )

    # Step 5 — build final result
    return FinalSubagentResult(
        score              = score,
        accepted           = accepted,
        summary            = llm_eval.summary,
        acceptance_reasons = reasons if accepted     else None,
        rejection_reasons  = reasons if not accepted else None,
    )
