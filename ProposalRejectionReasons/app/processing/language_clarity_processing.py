from schemas.language_clarity.language_clarity_eval_schema import LanguageClarityEvalSchema
import language_tool_python

LANGUAGE_CLARITY_THRESHOLD = 5.0

# Thresholds for text metrics
MIN_WORD_COUNT            = 50
MAX_AVG_SENTENCE_LENGTH   = 35   # words per sentence — above this is hard to read
MAX_GRAMMAR_ERRORS_MINOR  = 3    # <= 3 errors → minor issues, half penalty
_grammar_tool = language_tool_python.LanguageTool('en-US')


def calc_text_metrics(proposal_text: str) -> dict:
    """
    Calculate text metrics using pure code — no LLM needed.

    Metrics:
        - word_count            : total number of words
        - avg_sentence_length   : average words per sentence
        - length_score          : 1.0 if word_count >= MIN_WORD_COUNT, else 0.0
        - grammar_error_count   : number of grammar/spelling errors detected
        - grammar_score         : 1.0 if 0 errors, 0.5 if <= 3 errors, 0.0 if > 3

    Args:
        proposal_text: The freelancer's proposal text.

    Returns:
        dict with all text metrics.
    """
    # word and sentence counts
    words     = proposal_text.split()
    sentences = [
        s.strip()
        for s in proposal_text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    word_count          = len(words)
    avg_sentence_length = round(word_count / len(sentences), 2) if sentences else 0.0
    length_score        = 1.0 if word_count >= MIN_WORD_COUNT else 0.0

    # grammar and spelling check
    grammar_errors      = _grammar_tool.check(proposal_text)
    grammar_error_count = len(grammar_errors)

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


def calc_language_clarity_score(
    llm_eval    : LanguageClarityEvalSchema,
    text_metrics: dict
) -> float:
    """
    Compute the final language clarity score.

    Weights:
        is_clear                    → 3 points
        is_professional             → 3 points
        not has_misleading_phrasing → 2 points
        grammar_score               → 1 point   (0.0 / 0.5 / 1.0)
        length_score                → 1 point   (0.0 / 1.0)
        ────────────────────────────────────────
        Total                       → 10 points

    Returns:
        float score between 0.0 and 10.0
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

    return round(score, 2)


def build_language_clarity_reasons(
    llm_eval    : LanguageClarityEvalSchema,
    text_metrics: dict
) -> list[str]:
    """
    Build a list of specific, actionable rejection reasons from flags and metrics.
    Each reason maps directly to one failed check.
    This list is what the SuperAgent will consume to generate recommendations.

    Args:
        llm_eval     : Output of LanguageClarityEvaluator.
        text_metrics : Output of calc_text_metrics.

    Returns:
        List of specific rejection reason strings. Empty if everything passed.
    """
    reasons = []

    if not llm_eval.is_clear:
        reasons.append(
            "Your proposal is hard to follow. Use shorter, clearer sentences."
        )
    if not llm_eval.is_professional:
        reasons.append(
            "Your tone doesn't sound professional to the client."
        )
    if llm_eval.has_misleading_phrasing:
        reasons.append(
            "Your proposal contains vague or misleading statements. "
            "Avoid empty promises and back up your claims with specifics."
        )
    if text_metrics["grammar_error_count"] > MAX_GRAMMAR_ERRORS_MINOR:
        reasons.append(
            f"Your proposal has {text_metrics['grammar_error_count']} grammar/spelling errors. "
            "Proofread before submitting."
        )
    if text_metrics["word_count"] < MIN_WORD_COUNT:
        reasons.append(
            f"Your proposal is too short ({text_metrics['word_count']} words). "
            "Add more detail to be convincing."
        )

    return reasons


def calc_language_clarity_result(
    llm_eval     : LanguageClarityEvalSchema,
    proposal_text: str,
    threshold    : float = LANGUAGE_CLARITY_THRESHOLD
) -> dict:
    """
    Full processing pipeline:
        1. Compute text metrics via pure code (word count, sentence length, grammar)
        2. Compute final score from LLM flags + text metrics
        3. Build specific rejection reasons from flags
        4. Apply rule-based acceptance decision

    Args:
        llm_eval      : Output of LanguageClarityEvaluator.
        proposal_text : The freelancer's proposal (needed for text metrics).
        threshold     : Minimum passing score (default: 5.0).

    Returns:
        dict with all fields needed for the SuperAgent or rejection report.
    """
    # Step 1 — text metrics (pure code, no LLM)
    text_metrics = calc_text_metrics(proposal_text=proposal_text)

    # Step 2 — final score
    score = calc_language_clarity_score(
        llm_eval=llm_eval,
        text_metrics=text_metrics
    )

    # Step 3 — specific reasons from flags (used by SuperAgent)
    reasons = build_language_clarity_reasons(
        llm_eval=llm_eval,
        text_metrics=text_metrics
    )

    # Step 4 — rule-based acceptance decision
    accepted = score >= threshold

    rejection_reason = None
    if not accepted:
        rejection_reason = " ".join(reasons) if reasons else llm_eval.summary

    return {
        "score"                   : score,
        "confidence"              : llm_eval.confidence_score,
        "accepted"                : accepted,
        "is_clear"                : llm_eval.is_clear,
        "is_professional"         : llm_eval.is_professional,
        "has_misleading_phrasing" : llm_eval.has_misleading_phrasing,
        "word_count"              : text_metrics["word_count"],
        "avg_sentence_length"     : text_metrics["avg_sentence_length"],
        "grammar_error_count"     : text_metrics["grammar_error_count"],
        "grammar_score"           : text_metrics["grammar_score"],
        "reasons"                 : reasons,           # per-flag reasons → SuperAgent
        "rejection_reason"        : rejection_reason,  # combined string → rejection report
        "summary"                 : llm_eval.summary,  # LLM explanation → human readability
    }

