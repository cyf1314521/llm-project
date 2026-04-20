"""
SemEval 2026 Task 12 评测器。

官方评分规则：
- 1.0（Full Match）：P = G
- 0.5（Partial Match）：P 是 G 的非空真子集（且无错选）
- 0.0（Incorrect）：其余情况（包含错选或空预测）

最终分数为所有样本实例分数的平均值。
"""

import json
import logging
from typing import Dict, List, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self):
        # 样本级统计
        self.total = 0
        self.correct = 0
        self.partial = 0
        self.incorrect = 0

        # 选项级混淆统计（用于 P/R/F1）
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)

        self.option_stats = defaultdict(
            lambda: {"correct": 0, "total_selected": 0, "total_should_select": 0}
        )

        self.error_cases = []
        self.partial_cases = []

        # 按题型（单答案/多答案）拆分统计
        self.single_answer = defaultdict(int)
        self.multi_answer = defaultdict(int)
        # “信息不足/none of above”类题目统计
        self.insufficient_info_count = 0
        self.insufficient_info_correct = 0
        self.insufficient_info_partial = 0

        self.prediction_types = {
            "full_match": 0,
            "partial_match": 0,
            "empty_prediction": 0,
            "wrong_only": 0,
            "mixed_error": 0,
            "over_complete": 0,
        }

    def _calculate_instance_score(
        self, predicted: Set[str], ground_truth: Set[str]
    ) -> float:
        """
        基于官方规则计算单样本分数。
        """
        if not predicted:
            return 0.0
        if predicted == ground_truth:
            return 1.0
        if predicted < ground_truth:
            return 0.5
        return 0.0

    def _classify_prediction(
        self, predicted: Set[str], ground_truth: Set[str], score: float
    ) -> str:
        """将预测归类到更细粒度错误类型，便于误差分析。"""
        if score == 1.0:
            return "full_match"
        if score == 0.5:
            return "partial_match"

        if not predicted:
            return "empty_prediction"

        true_positives = predicted & ground_truth
        false_positives = predicted - ground_truth
        false_negatives = ground_truth - predicted

        if not true_positives:
            return "wrong_only"
        if not false_negatives and false_positives:
            return "over_complete"
        return "mixed_error"

    def update(
        self,
        predicted: Set[str],
        ground_truth: Set[str],
        event_id: str = "",
        prediction_text: str = "",
        event: str = "",
        options: List[str] = None,
    ):
        """更新一次样本评测结果及所有统计项。"""
        self.total += 1

        score = self._calculate_instance_score(predicted, ground_truth)
        prediction_type = self._classify_prediction(predicted, ground_truth, score)
        self.prediction_types[prediction_type] += 1

        if score == 1.0:
            self.correct += 1
        elif score == 0.5:
            self.partial += 1
            if event_id:
                self.partial_cases.append({
                    "id": event_id,
                    "event": event,
                    "predicted": sorted(predicted),
                    "ground_truth": sorted(ground_truth),
                    "missing": sorted(ground_truth - predicted),
                    "score": 0.5,
                    "prediction_type": prediction_type,
                })
        else:
            self.incorrect += 1
            if event_id:
                self.error_cases.append({
                    "id": event_id,
                    "event": event,
                    "predicted": sorted(predicted),
                    "ground_truth": sorted(ground_truth),
                    "false_positives": sorted(predicted - ground_truth),
                    "false_negatives": sorted(ground_truth - predicted),
                    "prediction_type": prediction_type,
                    "prediction_text": prediction_text,
                    "options": (
                        [
                            f"option_{label}: {opt}"
                            for label, opt in zip(["A", "B", "C", "D"], options)
                        ]
                        if options
                        else []
                    ),
                })

        # 选项级统计（用于后续 macro-f1 与 option matrix）
        for option in predicted | ground_truth:
            if option in predicted and option in ground_truth:
                self.true_positives[option] += 1
                self.option_stats[option]["correct"] += 1
            elif option in predicted:
                self.false_positives[option] += 1
            else:
                self.false_negatives[option] += 1

            if option in predicted:
                self.option_stats[option]["total_selected"] += 1
            if option in ground_truth:
                self.option_stats[option]["total_should_select"] += 1

        # 按题目答案数统计（单答案 vs 多答案）
        is_single = len(ground_truth) == 1
        stats = self.single_answer if is_single else self.multi_answer
        stats["count"] += 1
        stats["correct"] += score == 1.0
        stats["partial"] = stats.get("partial", 0) + (score == 0.5)

        # 检测“信息不足/none of above”选项命中情况
        if options:
            for i, opt in enumerate(options):
                if "insufficient" in opt.lower() or "none of" in opt.lower():
                    option_label = ["A", "B", "C", "D"][i]
                    if option_label in ground_truth:
                        self.insufficient_info_count += 1
                        if score == 1.0:
                            self.insufficient_info_correct += 1
                        elif score == 0.5 and option_label in predicted:
                            self.insufficient_info_partial += 1
                    break

    def get_official_score(self) -> float:
        """返回官方主指标分数。"""
        if self.total == 0:
            return 0.0
        return (1.0 * self.correct + 0.5 * self.partial) / self.total

    def get_accuracy(self) -> float:
        """返回严格准确率（只计 full match）。"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def get_macro_f1(self) -> float:
        """基于选项标签（A/B/C/D）计算宏平均 F1。"""
        all_options = (
            set(self.true_positives.keys())
            | set(self.false_positives.keys())
            | set(self.false_negatives.keys())
        )
        if not all_options:
            return 0.0

        f1_scores = []
        for option in all_options:
            tp = self.true_positives[option]
            fp = self.false_positives[option]
            fn = self.false_negatives[option]

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)

    def get_option_matrix(self) -> Dict[str, Dict[str, float]]:
        """返回每个选项标签的 Precision/Recall/F1。"""
        matrix = {}
        for option, stats in self.option_stats.items():
            prec = (
                stats["correct"] / stats["total_selected"]
                if stats["total_selected"] > 0
                else 0.0
            )
            rec = (
                stats["correct"] / stats["total_should_select"]
                if stats["total_should_select"] > 0
                else 0.0
            )
            f1 = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
            matrix[option] = {"precision": prec, "recall": rec, "f1": f1}
        return matrix

    def get_summary(self) -> Dict:
        """汇总所有核心指标，供控制台展示和 JSON 落盘。"""
        return {
            "total": self.total,
            "official_score": self.get_official_score(),
            "full_match": self.correct,
            "partial_match": self.partial,
            "incorrect": self.incorrect,
            "strict_accuracy": self.get_accuracy(),
            "macro_f1": self.get_macro_f1(),
            "prediction_types": self.prediction_types,
            "insufficient_info_count": self.insufficient_info_count,
            "insufficient_info_accuracy": (
                self.insufficient_info_correct / self.insufficient_info_count
                if self.insufficient_info_count > 0
                else 0.0
            ),
            "insufficient_info_partial": self.insufficient_info_partial,
            "single_answer_count": self.single_answer["count"],
            "single_answer_accuracy": (
                self.single_answer["correct"] / self.single_answer["count"]
                if self.single_answer["count"] > 0
                else 0.0
            ),
            "single_answer_partial": self.single_answer.get("partial", 0),
            "multi_answer_count": self.multi_answer["count"],
            "multi_answer_accuracy": (
                self.multi_answer["correct"] / self.multi_answer["count"]
                if self.multi_answer["count"] > 0
                else 0.0
            ),
            "multi_answer_partial": self.multi_answer.get("partial", 0),
            "option_stats": dict(self.option_stats),
            "option_matrix": self.get_option_matrix(),
            "error_count": len(self.error_cases),
            "partial_count": len(self.partial_cases),
        }

    def save_results(self, filepath: str, approach_name: str = "BaselineApproach"):
        """保存完整评测报告（summary + 错误样例 + 部分正确样例）。"""
        results = {
            "approach": approach_name,
            "summary": self.get_summary(),
            "error_cases": self.error_cases,
            "partial_cases": self.partial_cases,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to: %s", filepath)
