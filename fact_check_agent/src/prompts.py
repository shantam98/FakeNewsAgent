"""All LLM prompt templates for the Fact-Check Agent.

Centralised here so prompt changes can be version-controlled and re-evaluated
without touching agent logic. Each prompt has a version comment — bump it
whenever the structure changes and re-run benchmarks.
"""

# ── v2.0 — factual/counter-factual context claims pipeline ───────────────────

VERDICT_SYNTHESIS_PROMPT = """\
You are a fact-checker making a verdict. You have structured evidence for and against the claim.

CLAIM: {claim_text}

SOURCE CREDIBILITY CONTEXT:
{source_credibility_note}

EVIDENCE CONTEXT CLAIMS:
{context_claims_block}

INSTRUCTIONS:
1. Factual evidence supports the claim being true. Counter-factual evidence challenges it.
2. Weigh both sides: strong counter-factual evidence can outweigh weak factual evidence.
3. Memory evidence (prior verified claims) carries weight proportional to its stated confidence.
4. Identify the 2-3 most decisive pieces of evidence that drove your verdict.
5. Factor source credibility into confidence: low-credibility source → reduce confidence 10-20
   points; high credibility → maintain or increase. High bias std → apply extra scepticism.

VERDICT LABELS — choose exactly one:
- "supported"  : Factual evidence clearly confirms the claim; counter-factual is weak or absent.
- "refuted"    : Counter-factual evidence clearly contradicts the claim; factual is weak or absent.
- "misleading" : Evidence is mixed, partial, ambiguous, or the claim is exaggerated/decontextualised.

IMPORTANT RULES:
- Output exactly one of: supported, refuted, misleading
- When in doubt, choose "misleading"
- Confidence 50–70 for partial evidence; >80 only when evidence is direct and unambiguous

Return JSON:
{{
  "verdict": "supported|refuted|misleading",
  "confidence_score": <integer 0–100>,
  "bias_score": <float 0.0–1.0>,
  "reasoning": "<2-3 sentences weighing factual vs counter-factual evidence>",
  "key_evidence": ["<most decisive evidence snippet 1>", "<most decisive evidence snippet 2>"],
  "evidence_links": ["<url1>", "<url2>"]
}}
"""

# ── v1.0 — context claim agent prompts ───────────────────────────────────────

QUESTION_GENERATION_PROMPT = """\
You are a fact-check question strategist. Given a claim, generate the 3 most crucial questions
needed to verify it from two angles.

Factual questions: if answered with confirming evidence, they SUPPORT the claim being true.
Counter-factual questions: if answered with confirming evidence, they REFUTE or COMPLICATE the claim.

Make questions specific, independently answerable, and directly tied to the core assertion.

CLAIM: {claim_text}

Return JSON only — no explanation:
{{
  "factual": ["<question 1>", "<question 2>", "<question 3>"],
  "counter_factual": ["<question 4>", "<question 5>", "<question 6>"]
}}
"""

CONTEXT_COVERAGE_PROMPT = """\
You are checking whether a set of questions about a claim can be answered from the provided context.

CLAIM: {claim_text}

QUESTIONS:
{questions_block}

AVAILABLE CONTEXT:
{context_block}

For each question, determine if the context contains a direct or strongly implied answer.
If yes, quote the most relevant evidence (max 200 characters). If no, mark as unanswered.

Return JSON only:
{{
  "coverage": [
    {{"question": "<exact question text>", "answered": true, "evidence": "<short quote>"}},
    {{"question": "<exact question text>", "answered": false, "evidence": null}},
    ...
  ]
}}
"""

TAVILY_SUMMARY_PROMPT = """\
You are extracting a single relevant claim from search results to help verify a fact-check question.

ORIGINAL CLAIM: {claim_text}
QUESTION: {question}

SEARCH RESULTS:
{search_results}

Summarise ONLY information that directly addresses the question and is relevant to the original claim.
Write 1-2 concise sentences as a factual statement. If the search results contain no relevant
information, respond with null.

Return JSON only:
{{"summary": "<1-2 sentence factual statement>" | null}}
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
