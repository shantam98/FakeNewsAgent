"""Tests for the freshness tool — no OpenAI API key required."""
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from fact_check_agent.src.tools.freshness_tool import check_freshness


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_openai_response(revalidate: bool, reason: str = "test reason", category: str = "historical"):
    """Return a mock OpenAI ChatCompletion for the freshness check."""
    content = json.dumps({
        "revalidate":     revalidate,
        "reason":         reason,
        "claim_category": category,
    })
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def recent(days: int = 1) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


# ── Fresh (revalidate=False) paths ────────────────────────────────────────────

def test_historical_claim_fresh():
    """Historical facts should not need revalidation even if verified a year ago."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=False, reason="Historical fact — no revalidation needed.", category="historical"
        )
        result = check_freshness(
            claim_text         = "The French Revolution began in 1789.",
            verdict_label      = "supported",
            verdict_confidence = 0.99,
            last_verified_at   = recent(days=365),
            api_key            = "fake-key",
            model              = "gpt-4o",
        )

    assert result["revalidate"] is False
    assert result["claim_category"] == "historical"
    assert result["reason"] != ""


def test_scientific_claim_verified_recently():
    """Scientific consensus verified 30 days ago should be fresh (threshold = 180 days)."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=False, category="scientific"
        )
        result = check_freshness(
            claim_text         = "Vaccines do not cause autism.",
            verdict_label      = "supported",
            verdict_confidence = 0.98,
            last_verified_at   = recent(days=30),
            api_key            = "fake-key",
            model              = "gpt-4o",
        )

    assert result["revalidate"] is False


# ── Stale (revalidate=True) paths ─────────────────────────────────────────────

def test_political_claim_stale():
    """Political claims verified 10 days ago should need revalidation (threshold = 7 days)."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=True, reason="Political claim is 10 days old.", category="political"
        )
        result = check_freshness(
            claim_text         = "The prime minister announced a new policy yesterday.",
            verdict_label      = "supported",
            verdict_confidence = 0.85,
            last_verified_at   = recent(days=10),
            api_key            = "fake-key",
            model              = "gpt-4o",
        )

    assert result["revalidate"] is True
    assert result["claim_category"] == "political"


def test_ongoing_event_stale():
    """Ongoing events verified 5 days ago should need revalidation (threshold = 3 days)."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=True, category="ongoing_event"
        )
        result = check_freshness(
            claim_text         = "The ceasefire negotiations are ongoing.",
            verdict_label      = "supported",
            verdict_confidence = 0.70,
            last_verified_at   = recent(days=5),
            api_key            = "fake-key",
            model              = "gpt-4o",
        )

    assert result["revalidate"] is True


# ── Return value contract ─────────────────────────────────────────────────────

def test_returns_all_keys():
    """Result must always contain revalidate, reason, and claim_category."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=False, reason="Fresh.", category="economic"
        )
        result = check_freshness(
            claim_text         = "Unemployment fell to 3.8%.",
            verdict_label      = "supported",
            verdict_confidence = 0.80,
            last_verified_at   = recent(days=2),
            api_key            = "fake-key",
            model              = "gpt-4o",
        )

    assert "revalidate"     in result
    assert "reason"         in result
    assert "claim_category" in result
    assert isinstance(result["revalidate"], bool)


def test_revalidate_is_bool_not_string():
    """revalidate must be a Python bool — not a JSON string "true"."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=True
        )
        result = check_freshness(
            claim_text="Any claim.", verdict_label="refuted",
            verdict_confidence=0.5, last_verified_at=recent(days=1),
            api_key="fake-key", model="gpt-4o",
        )

    assert result["revalidate"] is True   # strict identity, not just truthiness


# ── Timezone handling ─────────────────────────────────────────────────────────

def test_naive_datetime_handled():
    """last_verified_at without tzinfo should not raise — tool must normalise it."""
    naive_ts = datetime.utcnow() - timedelta(days=2)
    assert naive_ts.tzinfo is None  # confirm it's naive

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            revalidate=False
        )
        # Should not raise
        result = check_freshness(
            claim_text="Claim.", verdict_label="supported",
            verdict_confidence=0.9, last_verified_at=naive_ts,
            api_key="fake-key", model="gpt-4o",
        )

    assert "revalidate" in result


# ── Failure / degraded mode ───────────────────────────────────────────────────

def test_api_failure_defaults_to_revalidate():
    """If OpenAI call fails, tool must default to revalidate=True (safe fallback)."""
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = Exception("timeout")
        result = check_freshness(
            claim_text="Any claim.", verdict_label="supported",
            verdict_confidence=0.9, last_verified_at=recent(days=1),
            api_key="fake-key", model="gpt-4o",
        )

    assert result["revalidate"] is True
    assert result["claim_category"] == "unknown"


def test_invalid_json_response_defaults_to_revalidate():
    """If LLM returns non-JSON, tool must not crash and must revalidate."""
    choice = MagicMock()
    choice.message.content = "I cannot determine freshness."  # not JSON
    resp = MagicMock()
    resp.choices = [choice]

    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = resp
        result = check_freshness(
            claim_text="Any claim.", verdict_label="supported",
            verdict_confidence=0.9, last_verified_at=recent(days=1),
            api_key="fake-key", model="gpt-4o",
        )

    assert result["revalidate"] is True
