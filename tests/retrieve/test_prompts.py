from fakes import FakeLLM

from retrieve.prompts import (
    REFUSAL_MARKER,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_refusal,
)


def test_system_prompt_instructs_the_refusal_marker():
    assert REFUSAL_MARKER in SYSTEM_PROMPT


def test_system_prompt_instructs_the_citation_format():
    assert "[title, HH:MM:SS]" in SYSTEM_PROMPT


def test_build_user_prompt_includes_query_and_context():
    prompt = build_user_prompt(
        "What did they decide about the budget?",
        ["[Q1 Planning, 00:04:12]\nWe agreed to cut travel spend."],
    )

    assert "What did they decide about the budget?" in prompt
    assert "[Q1 Planning, 00:04:12]" in prompt
    assert "We agreed to cut travel spend." in prompt


def test_build_user_prompt_joins_multiple_context_blocks():
    prompt = build_user_prompt("query", ["first excerpt", "second excerpt"])

    assert "first excerpt" in prompt
    assert "second excerpt" in prompt
    assert prompt.index("first excerpt") < prompt.index("second excerpt")


def test_is_refusal_true_for_exact_marker():
    assert is_refusal(REFUSAL_MARKER) is True


def test_is_refusal_tolerates_surrounding_whitespace():
    assert is_refusal(f"  {REFUSAL_MARKER}\n") is True


def test_is_refusal_false_for_a_real_answer():
    assert is_refusal("The budget was cut by 10% [Q1 Planning, 00:04:12].") is False


def test_fake_llm_refusal_round_trips_through_is_refusal():
    llm = FakeLLM(response=REFUSAL_MARKER)

    output = llm.complete(SYSTEM_PROMPT, build_user_prompt("unrelated question", []))

    assert is_refusal(output) is True
    assert llm.calls == [(SYSTEM_PROMPT, build_user_prompt("unrelated question", []))]


def test_fake_llm_answer_is_not_a_refusal():
    llm = FakeLLM(response="Yes, per [Q1 Planning, 00:04:12].")

    output = llm.complete(SYSTEM_PROMPT, build_user_prompt("query", ["ctx"]))

    assert is_refusal(output) is False
