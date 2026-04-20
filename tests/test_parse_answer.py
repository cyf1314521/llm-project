"""答案解析函数（parse_answer）的单元测试。"""

import pytest
from src.approaches import parse_answer


class TestParseAnswer:
    """parse_answer 基础功能测试。"""

    def test_single_answer(self):
        text = "Some reasoning.\nFinal Answer I Reasoned: A"
        assert parse_answer(text) == {"A"}

    def test_multiple_answers(self):
        text = "Analysis...\nFinal Answer I Reasoned: A,B,C"
        assert parse_answer(text) == {"A", "B", "C"}

    def test_multiple_answers_with_spaces(self):
        text = "Final Answer I Reasoned: A, B, C"
        assert parse_answer(text) == {"A", "B", "C"}

    def test_case_insensitive(self):
        text = "final answer i reasoned: a,b"
        assert parse_answer(text) == {"A", "B"}

    def test_last_match_wins(self):
        """When the model restates the format, take the last occurrence."""
        text = (
            "Final Answer I Reasoned: A,B\n"
            "Wait, let me reconsider...\n"
            "Final Answer I Reasoned: B"
        )
        assert parse_answer(text) == {"B"}

    def test_empty_string(self):
        assert parse_answer("") == set()

    def test_none_input(self):
        assert parse_answer(None) == set()

    def test_no_match_returns_empty(self):
        text = "I think the answer is probably A but I'm not sure."
        assert parse_answer(text) == set()

    def test_fallback_with_answer_keyword(self):
        text = "Based on the evidence above,\nAnswer: B"
        assert parse_answer(text) == {"B"}

    def test_no_false_positive_on_english_article(self):
        """Should NOT match 'A' in 'A man fired twice'."""
        text = "A man fired twice at the target. Based on this, C is the cause."
        result = parse_answer(text)
        assert "A" not in result

    def test_valid_options_only(self):
        text = "Final Answer I Reasoned: A,E,B"
        assert parse_answer(text) == {"A", "B"}

    def test_whitespace_around_answer(self):
        text = "Final Answer I Reasoned:   B  "
        assert parse_answer(text) == {"B"}


class TestParseAnswerEdgeCases:
    """边界场景与回归测试。"""

    def test_answer_in_middle_of_text(self):
        text = (
            "Let me analyze...\n"
            "Final Answer I Reasoned: D\n"
        )
        assert parse_answer(text) == {"D"}

    def test_all_four_options(self):
        text = "Final Answer I Reasoned: A,B,C,D"
        assert parse_answer(text) == {"A", "B", "C", "D"}

    def test_repeated_option(self):
        text = "Final Answer I Reasoned: A,A,B"
        assert parse_answer(text) == {"A", "B"}
