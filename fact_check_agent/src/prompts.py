"""All LLM prompt templates for the Fact-Check Agent.

Centralised here so prompt changes can be version-controlled and re-evaluated
without touching agent logic. Each prompt has a version comment — bump it
whenever the structure changes and re-run benchmarks.
"""

# ── v1.1 ─────────────────────────────────────────────────────────────────────

VERDICT_SYNTHESIS_PROMPT = """\
You are a fact-checker. Analyse the claim against the provided evidence and return a verdict.

CLAIM: {claim_text}

EVIDENCE:
{evidence_block}

SOURCE CREDIBILITY CONTEXT:
{source_credibility_note}

The source credibility score reflects how often this source's past claims (on similar topics)
have been verified as true. A low score does not make the current claim false — weigh it
alongside the evidence. High bias std means the source is inconsistent across topics.

VERDICT LABELS — you MUST choose exactly one:
- "supported"  : The evidence directly and clearly confirms the claim is true.
- "refuted"    : The evidence directly and clearly contradicts or disproves the claim.
- "misleading" : Use this in ALL other cases, including:
                   • The evidence is insufficient, ambiguous, or only partially addresses the claim
                   • The claim is exaggerated, taken out of context, or missing key caveats
                   • The provided document does not contain enough information to confirm or deny

IMPORTANT RULES:
- You MUST output exactly one of the three strings: supported, refuted, misleading
- Do NOT output "not supported", "unverified", "not enough information", or any other label
- When in doubt, choose "misleading" — it is the correct label for uncertain or partial evidence
- Only choose "supported" or "refuted" when the evidence is direct, explicit, and unambiguous
- Confidence score: use 50–70 when evidence is partial; only use >80 when evidence is clear and direct

Return a JSON object with exactly these keys:
{{
  "verdict": "supported|refuted|misleading",
  "confidence_score": <integer 0–100>,
  "bias_score": <float 0.0–1.0>,
  "reasoning": "<2–3 sentence explanation>",
  "evidence_links": ["<url1>", "<url2>"]
}}
"""

CROSS_MODAL_PROMPT = """\
You are checking whether a news image is being used in a misleading context.

[CLAIM TEXT]
{claim_text}

[IMAGE CAPTION]
{image_caption}

Task: Identify only clear logical conflicts between what the claim states and what the image \
objectively shows. Do NOT flag:
- Stylistic mismatches or tone differences
- Missing information (absence of evidence is not conflict)
- Speculative or inferred connections

If there is a clear, direct logical conflict, explain it in one concise sentence.

Return JSON:
{{
  "conflict": true | false,
  "explanation": "<one sentence describing the conflict, or null if no conflict>"
}}
"""

# ── v1.1 vision cross-modal (Gemma 4 / Ollama) ───────────────────────────────

CROSS_MODAL_VISION_PROMPT = """\
You are checking whether a news image is being used in a misleading context.

[CLAIM TEXT]
{claim_text}

Look at the image above carefully. Identify only clear logical conflicts between what the \
claim states and what the image objectively shows. Do NOT flag:
- Stylistic mismatches or tone differences
- Missing information (absence of evidence is not conflict)
- Speculative or inferred connections

If there is a clear, direct logical conflict, explain it in one concise sentence.

Return JSON only — no markdown fences:
{{
  "conflict": true | false,
  "explanation": "<one sentence describing the conflict, or null if no conflict>"
}}
"""

# ── v1.0 freshness classifier ────────────────────────────────────────────────

FRESHNESS_CHECK_PROMPT = """\
You are deciding whether a cached fact-check verdict needs live re-verification.

CLAIM: {claim_text}

PRIOR VERDICT: {verdict_label} ({verdict_confidence:.0%} confidence)
LAST VERIFIED: {time_since_verified_days} days ago

Guidelines for re-verification:
- Political claims, election results, government policy: re-verify if > 7 days old
- Ongoing events (court cases, conflicts, legislation in progress): re-verify if > 3 days old
- Economic data (prices, unemployment, GDP): re-verify if > 14 days old
- Scientific consensus, medical guidelines: re-verify if > 180 days old
- Historical facts, geography, physical constants: almost never need re-verification
- Satire or clearly fabricated claims: rarely need re-verification

Return a JSON object:
{{
  "revalidate": true | false,
  "reason": "<one sentence explaining the decision>",
  "claim_category": "<political|ongoing_event|economic|scientific|historical|other>"
}}
"""

# ── SOTA prompts (not used in baseline — wired in when SOTA flags enabled) ───

IS_RETRIEVAL_NEEDED_PROMPT = """\
Given the following claim, decide whether external evidence is needed to verify it,
or whether it is self-evidently true or false without retrieval.

CLAIM: {claim_text}

Answer with a JSON object:
{{
  "retrieval_needed": true | false,
  "reason": "<one sentence>"
}}
"""

CHUNK_RELEVANCE_PROMPT = """\
Rate each retrieved text chunk for relevance to the following claim on a scale of 1–5.
Exclude chunks rated below 3 from evidence synthesis.

CLAIM: {claim_text}

CHUNKS:
{chunks_block}

Return a JSON array:
[
  {{"index": 0, "relevance": <1–5>, "keep": true | false}},
  ...
]
"""

DECOMPOSITION_PROMPT = """\
Decompose the following compound claim into the smallest independently falsifiable sub-claims.

CLAIM: {claim_text}

Return JSON:
{{
  "sub_claims": [
    {{"text": "<atomic sub-claim>", "verifiable": true | false}}
  ]
}}

If the claim is already atomic, return a single-element list with verifiable: true.
"""

ADVOCATE_PROMPT = """\
You are arguing that the following claim is {position}.
Present the strongest {position_adj} evidence and reasoning you can find.
Be concise — 3–5 bullet points.

CLAIM: {claim_text}

AVAILABLE EVIDENCE:
{evidence_block}
"""

ARBITER_PROMPT = """\
Two agents have argued opposing positions on the following claim.
Synthesise their arguments and produce a final ruling.

CLAIM: {claim_text}

ARGUMENT FOR SUPPORTED:
{argument_for}

ARGUMENT FOR REFUTED:
{argument_against}

You MUST choose exactly one verdict: "supported", "refuted", or "misleading".
Do NOT use any other label. If neither argument is convincing, choose "misleading".

Return JSON with the same structure as the verdict synthesis prompt:
{{
  "verdict": "supported|refuted|misleading",
  "confidence_score": <integer 0–100>,
  "bias_score": <float 0.0–1.0>,
  "reasoning": "<2–3 sentences>",
  "evidence_links": []
}}
"""
