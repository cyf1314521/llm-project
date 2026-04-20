"""评测器（Evaluator）评分逻辑测试。"""

import pytest
from src.evaluator import Evaluator


class TestInstanceScore:
    """_calculate_instance_score 的测试。"""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_full_match_single(self):
        score = self.evaluator._calculate_instance_score({"A"}, {"A"})
        assert score == 1.0

    def test_full_match_multi(self):
        score = self.evaluator._calculate_instance_score({"A", "B"}, {"A", "B"})
        assert score == 1.0

    def test_partial_match_subset(self):
        score = self.evaluator._calculate_instance_score({"A"}, {"A", "B"})
        assert score == 0.5

    def test_empty_prediction(self):
        score = self.evaluator._calculate_instance_score(set(), {"A"})
        assert score == 0.0

    def test_wrong_selection(self):
        score = self.evaluator._calculate_instance_score({"B"}, {"A"})
        assert score == 0.0

    def test_superset_is_wrong(self):
        """Predicting extra options beyond ground truth = 0 points."""
        score = self.evaluator._calculate_instance_score({"A", "B"}, {"A"})
        assert score == 0.0

    def test_mixed_correct_and_wrong(self):
        score = self.evaluator._calculate_instance_score({"A", "C"}, {"A", "B"})
        assert score == 0.0


class TestOfficialScore:
    """官方总分聚合逻辑测试。"""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_all_correct(self):
        self.evaluator.update({"A"}, {"A"}, event_id="1")
        self.evaluator.update({"B"}, {"B"}, event_id="2")
        assert self.evaluator.get_official_score() == 1.0

    def test_all_wrong(self):
        self.evaluator.update({"B"}, {"A"}, event_id="1")
        self.evaluator.update({"C"}, {"D"}, event_id="2")
        assert self.evaluator.get_official_score() == 0.0

    def test_mixed_scores(self):
        self.evaluator.update({"A"}, {"A"}, event_id="1")       # 1.0
        self.evaluator.update({"A"}, {"A", "B"}, event_id="2")  # 0.5
        self.evaluator.update({"C"}, {"A"}, event_id="3")       # 0.0
        expected = (1.0 + 0.5 + 0.0) / 3
        assert abs(self.evaluator.get_official_score() - expected) < 1e-6

    def test_empty_evaluator(self):
        assert self.evaluator.get_official_score() == 0.0


class TestPredictionClassification:
    """预测类型分类逻辑测试。"""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_full_match(self):
        t = self.evaluator._classify_prediction({"A"}, {"A"}, 1.0)
        assert t == "full_match"

    def test_partial_match(self):
        t = self.evaluator._classify_prediction({"A"}, {"A", "B"}, 0.5)
        assert t == "partial_match"

    def test_empty_prediction(self):
        t = self.evaluator._classify_prediction(set(), {"A"}, 0.0)
        assert t == "empty_prediction"

    def test_wrong_only(self):
        t = self.evaluator._classify_prediction({"C"}, {"A"}, 0.0)
        assert t == "wrong_only"

    def test_over_complete(self):
        t = self.evaluator._classify_prediction({"A", "C"}, {"A"}, 0.0)
        assert t == "over_complete"

    def test_mixed_error(self):
        t = self.evaluator._classify_prediction({"A", "C"}, {"A", "B"}, 0.0)
        assert t == "mixed_error"


class TestSummary:
    """get_summary 输出结构测试。"""

    def test_summary_keys(self):
        evaluator = Evaluator()
        evaluator.update({"A"}, {"A"}, event_id="1")
        summary = evaluator.get_summary()

        required_keys = [
            "total", "official_score", "full_match", "partial_match",
            "incorrect", "strict_accuracy", "macro_f1", "prediction_types",
            "option_matrix",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"
