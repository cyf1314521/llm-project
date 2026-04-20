"""后处理工具函数单元测试。"""

import pytest
from src.approaches import (
    detect_duplicate_options,
    find_none_correct_option,
    post_process_answers,
)


class TestDetectDuplicateOptions:

    def test_no_duplicates(self):
        options = ["Cause A", "Cause B", "Cause C", "Cause D"]
        assert detect_duplicate_options(options) == []

    def test_identical_options(self):
        options = ["Storm hit Texas", "Cause B", "Storm hit Texas", "Cause D"]
        result = detect_duplicate_options(options)
        assert len(result) == 1
        assert result[0] == ("A", "C", "identical")

    def test_case_insensitive(self):
        options = ["Storm Hit Texas", "Cause B", "storm hit texas", "Cause D"]
        result = detect_duplicate_options(options)
        assert len(result) == 1

    def test_whitespace_handling(self):
        options = [" Storm ", "Cause B", "Storm", "Cause D"]
        result = detect_duplicate_options(options)
        assert len(result) == 1


class TestFindNoneCorrectOption:

    def test_standard_none_option(self):
        options = ["Cause A", "Cause B", "None of the others are correct causes.", "Cause D"]
        assert find_none_correct_option(options) == "C"

    def test_none_of_the_above(self):
        options = ["Cause A", "None of the above", "Cause C", "Cause D"]
        assert find_none_correct_option(options) == "B"

    def test_no_none_option(self):
        options = ["Cause A", "Cause B", "Cause C", "Cause D"]
        assert find_none_correct_option(options) is None

    def test_case_insensitive(self):
        options = ["NONE OF THE OTHERS are correct", "B", "C", "D"]
        assert find_none_correct_option(options) == "A"


class TestPostProcessAnswers:

    def test_no_changes_needed(self):
        options = ["Cause A", "Cause B", "Cause C", "Cause D"]
        result = post_process_answers({"A"}, options)
        assert result == {"A"}

    def test_duplicate_option_expansion(self):
        """If A and C are identical and only A is selected, both should be."""
        options = ["Storm hit", "Cause B", "Storm hit", "Cause D"]
        result = post_process_answers({"A"}, options)
        assert result == {"A", "C"}

    def test_none_with_others_removes_none(self):
        """'None correct' should be removed if other options are selected."""
        options = ["Cause A", "Cause B", "None of the others are correct.", "Cause D"]
        result = post_process_answers({"A", "C"}, options)
        assert result == {"A"}

    def test_none_alone_stays(self):
        """'None correct' selected alone should stay."""
        options = ["Cause A", "Cause B", "None of the others are correct.", "Cause D"]
        result = post_process_answers({"C"}, options)
        assert result == {"C"}

    def test_empty_answers_unchanged(self):
        options = ["A", "B", "C", "D"]
        result = post_process_answers(set(), options)
        assert result == set()
