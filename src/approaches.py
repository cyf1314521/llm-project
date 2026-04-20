"""
SemEval 2026 Task 12 的推理策略集合。

评分机制决定了策略偏好：
- 1.0：预测集合与真值完全一致
- 0.5：预测是非空真子集（不能有错选）
- 0.0：其余情况（含错选或空选）

因此整体设计偏向“保守高精度”。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Any
import re
import logging
import json
import os
from datetime import datetime

from src.llm import BaseLLM
from src.retriever import DocumentRetriever
from src.dataloader import AERItem
from src.prompts import PROMPTS

logger = logging.getLogger(__name__)


def parse_answer(response: str) -> set:
    """
    从模型输出文本中提取最终选项集合。

    这是项目统一答案解析入口（single source of truth）。
    通过匹配最后一次出现的
    "Final Answer I Reasoned: ..."
    来避免模型在中间推理时重复该格式导致误解析。
    """
    if not response:
        return set()

    pattern = r"Final Answer I Reasoned:\s*([A-Z](?:\s*,\s*[A-Z])*)"
    matches = re.findall(pattern, response, re.IGNORECASE)

    if matches:
        answer_str = matches[-1].strip()
        answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
        return {a for a in answers if a in ["A", "B", "C", "D"]}

    # 兜底规则：尝试在结尾处匹配 Answer/Select/Choose 格式
    fallback = r"(?:answer|select|choose)[:\s]+([A-D](?:\s*,\s*[A-D])*)\s*$"
    match = re.search(fallback, response[-300:], re.IGNORECASE | re.MULTILINE)
    if match:
        answers = [a.strip().upper() for a in match.group(1).split(",") if a.strip()]
        return {a for a in answers if a in ["A", "B", "C", "D"]}

    return set()


# ============================================================
# 后处理工具：用于约束答案逻辑一致性
# ============================================================

def detect_duplicate_options(options: list) -> list:
    """
    检测完全相同的选项文本（重复选项）。

    返回：
        [(label1, label2, "identical"), ...]
    """
    labels = ["A", "B", "C", "D"]
    duplicates = []

    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            opt_i = options[i].strip().lower()
            opt_j = options[j].strip().lower()

            if opt_i == opt_j:
                duplicates.append((labels[i], labels[j], "identical"))

    return duplicates


def find_none_correct_option(options: list) -> str:
    """
    检测“以上都不是正确原因”类选项。

    返回：
        对应选项标签（A/B/C/D）或 None
    """
    labels = ["A", "B", "C", "D"]
    none_keywords = ["none of the others", "none of the above", "none are correct"]

    for i, opt in enumerate(options):
        opt_lower = opt.lower()
        if any(keyword in opt_lower for keyword in none_keywords):
            return labels[i]

    return None


def post_process_answers(answers: set, options: list) -> set:
    """
    对模型答案进行后处理，保证基本逻辑一致性。

    规则：
    1) 重复选项要么同时选，要么同时不选
    2) “None correct” 不能与其他选项并存
    3) 不在此处强制补全空答案，留给上层策略控制
    """
    if not answers:
        return answers

    processed = answers.copy()

    # Rule 1: Handle duplicate options
    duplicates = detect_duplicate_options(options)
    for label1, label2, dup_type in duplicates:
        if label1 in processed or label2 in processed:
            processed.add(label1)
            processed.add(label2)

    # Rule 2: Handle mutual exclusivity of "None correct"
    none_label = find_none_correct_option(options)
    if none_label and none_label in processed:
        other_answers = processed - {none_label}
        if other_answers:
            processed.discard(none_label)

    return processed


# ============================================================
# 策略基类：封装公共流程（检索、格式化、组装消息、后处理）
# ============================================================

class BaseApproach(ABC):
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def solve(self, item: AERItem, prompt_name: str) -> str:
        pass

    def _retrieve_documents(self, item: AERItem) -> list:
        """检索相关文档；若未配置检索器则返回全量文档。"""
        if self.retriever:
            return self.retriever.retrieve(
                item.event, item.title_snippet, item.documents, item.options
            )
        return item.documents

    def _format_context(self, item: AERItem, documents: list) -> Tuple[str, str]:
        """将文档与选项格式化为可插入 Prompt 的文本块。"""
        docs_text = "\n".join(
            f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents)
        )
        options_text = "\n".join(
            f"{label}: {opt}"
            for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        return docs_text, options_text

    def _build_messages(self, item: AERItem, prompt_name: str) -> list:
        """构建一次标准 LLM 调用所需的 messages。"""
        documents = self._retrieve_documents(item)
        docs_text, options_text = self._format_context(item, documents)

        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _apply_post_processing(self, response: str, item: AERItem) -> str:
        """解析并后处理答案；若变化则在输出末尾重写最终答案。"""
        raw_answers = parse_answer(response)
        processed_answers = post_process_answers(raw_answers, item.options)

        if processed_answers != raw_answers:
            response += (
                f"\n\n[Post-processing applied: {sorted(raw_answers)} "
                f"-> {sorted(processed_answers)}]"
            )
            if processed_answers:
                response += (
                    f"\nFinal Answer I Reasoned: "
                    f"{','.join(sorted(processed_answers))}"
                )

        return response


# ============================================================
# Baseline：单轮 CoT 推理
# ============================================================

class BaselineApproach(BaseApproach):
    """零样本 Chain-of-Thought 单轮推理。"""

    def solve(self, item: AERItem, prompt_name: str = "cot") -> str:
        messages = self._build_messages(item, prompt_name)
        response = self.llm.generate(messages)
        return self._apply_post_processing(response, item)


# ============================================================
# Conservative：偏保守策略（优先减少错选）
# ============================================================

class ConservativeApproach(BaseApproach):
    """
    面向官方评分的保守策略。

    核心思想：宁可漏选，也尽量避免错选。
    - 错选：0 分
    - 正确子集：0.5 分
    """

    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        messages = self._build_messages(item, prompt_name)
        response = self.llm.generate(messages)
        return self._apply_post_processing(response, item)


# ============================================================
# Lightweight Self-Consistency：轻量投票版
# ============================================================

class LightweightConsistencyApproach(BaseApproach):
    """
    轻量 Self-Consistency，使用“选项级投票”而不是“答案集合投票”。
    默认采样 3 次，超过阈值的选项被保留。
    """

    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 3
        self.temperature = 0.5
        self.vote_threshold = 2

    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        messages = self._build_messages(item, prompt_name)

        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []

        for i in range(self.num_samples):
            response = self.llm.generate(
                messages, temperature=self.temperature
            )
            all_responses.append(response)

            answers = parse_answer(response)
            for opt in answers:
                option_votes[opt] += 1

        # 仅保留票数达到阈值的选项
        voted_answers = {
            opt for opt, count in option_votes.items()
            if count >= self.vote_threshold
        }

        # 若没有选项过阈值，则回退到“最高票选项”
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {
                    opt for opt, count in option_votes.items()
                    if count == max_votes
                }

        final_answers = post_process_answers(voted_answers, item.options)
        vote_summary = ", ".join(
            f"{opt}:{count}" for opt, count in sorted(option_votes.items())
        )

        output = (
            f"========== LIGHTWEIGHT CONSISTENCY ==========\n"
            f"Samples: {self.num_samples}, Threshold: {self.vote_threshold}\n"
            f"Vote counts: {vote_summary}\n"
            f"Voted answers: {sorted(final_answers)}\n\n"
            f"========== BEST RESPONSE ==========\n"
            f"{all_responses[0] if all_responses else 'No response'}\n\n"
            f"Final Answer I Reasoned: "
            f"{','.join(sorted(final_answers)) if final_answers else 'A'}"
        )
        return output


# ============================================================
# Two-Pass：两阶段策略（先召回候选，再严格验证）
# ============================================================

class TwoPassApproach(BaseApproach):
    """
    双阶段推理，明确拆成两次模型调用：
    - Pass1：宽松召回候选（偏高召回）
    - Pass2：严格因果验证（偏高精度）
    """

    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        documents = self._retrieve_documents(item)
        docs_text, options_text = self._format_context(item, documents)

        # ============ PASS 1：宽松候选筛选 ============
        pass1_system = (
            "You are an expert in causal reasoning. Your task is to "
            "identify ALL potentially relevant options."
        )
        pass1_user = (
            f"TARGET EVENT: {item.event}\n\n"
            f"DOCUMENTS:\n{docs_text}\n\n"
            f"OPTIONS:\n{options_text}\n\n"
            f"TASK: For each option, determine if it has ANY potential "
            f"connection to the target event.\n"
            f"Be INCLUSIVE at this stage - mark as CANDIDATE if there's "
            f"any possible relationship.\n\n"
            f"For each option, answer:\n"
            f"- Option A: CANDIDATE or REJECT? (one word)\n"
            f"- Option B: CANDIDATE or REJECT? (one word)\n"
            f"- Option C: CANDIDATE or REJECT? (one word)\n"
            f"- Option D: CANDIDATE or REJECT? (one word)\n\n"
            f"Then list all CANDIDATE options."
        )

        pass1_response = self.llm.generate(
            [
                {"role": "system", "content": pass1_system},
                {"role": "user", "content": pass1_user},
            ],
            temperature=0.3,
            top_p=0.9,
        )

        # 解析 Pass 1 的候选结果
        candidates = set()
        for label in ["A", "B", "C", "D"]:
            if re.search(
                rf"Option {label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE
            ):
                candidates.add(label)
            elif re.search(
                rf"{label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE
            ):
                candidates.add(label)

        if not candidates:
            match = re.search(
                r"candidates?[:\s]*([A-D](?:\s*,\s*[A-D])*)",
                pass1_response,
                re.IGNORECASE,
            )
            if match:
                candidates = {
                    c.strip().upper()
                    for c in match.group(1).split(",")
                    if c.strip().upper() in ["A", "B", "C", "D"]
                }

        if not candidates:
            candidates = {"A", "B", "C", "D"}

        # ============ PASS 2：严格验证 ============
        candidates_text = ", ".join(sorted(candidates))
        pass2_system = (
            "You are an expert in causal reasoning. Your task is to verify "
            "which candidates are TRUE CAUSES.\n\n"
            "CRITICAL SCORING RULE:\n"
            "- Selecting ANY wrong option = 0 points\n"
            "- Missing some correct options = 0.5 points\n"
            "- Be CONSERVATIVE: Only select options you are CERTAIN about."
        )

        pass2_user = (
            f"TARGET EVENT: {item.event}\n\n"
            f"DOCUMENTS:\n{docs_text}\n\n"
            f"CANDIDATE OPTIONS (from Pass 1): {candidates_text}\n\n"
            f"For each candidate, verify:\n"
            f"1. TEMPORAL: Does evidence show this happened BEFORE the "
            f"target event? (YES/NO)\n"
            f"2. CAUSAL: Is there a clear mechanism by which this CAUSED "
            f"the event? (YES/NO)\n"
            f"3. EVIDENCE: Is there direct documentary support? (YES/NO)\n\n"
            f"Only select options with ALL THREE = YES.\n\n"
            f"Remember: Wrong selection = 0 points. Be conservative!\n\n"
            f"Final Answer I Reasoned: [Only verified options]"
        )

        pass2_response = self.llm.generate(
            [
                {"role": "system", "content": pass2_system},
                {"role": "user", "content": pass2_user},
            ],
            temperature=0.1,
            top_p=1,
        )

        raw_answers = parse_answer(pass2_response)
        final_answers = post_process_answers(raw_answers, item.options)

        output = (
            f"========== TWO-PASS APPROACH ==========\n\n"
            f"----- PASS 1: Candidate Selection -----\n"
            f"Candidates identified: {sorted(candidates)}\n\n"
            f"{pass1_response}\n\n"
            f"----- PASS 2: Strict Verification -----\n"
            f"{pass2_response}\n\n"
            f"----- POST-PROCESSING -----\n"
            f"Raw answers: {sorted(raw_answers)}\n"
            f"Final answers: {sorted(final_answers)}\n\n"
            f"Final Answer I Reasoned: "
            f"{','.join(sorted(final_answers)) if final_answers else 'A'}"
        )
        return output


# ============================================================
# Self-Consistency Refinement：高采样投票 + 规则修正
# ============================================================

class SelfConsistencyRefinementApproach(BaseApproach):
    """
    结合多次采样一致性 + 选项级投票 + 后处理修正。
    """

    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 7
        self.temperature = 0.5
        self.top_p = 0.95
        self.vote_threshold = 4
        self.d_option_threshold = 5

    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        messages = self._build_messages(item, prompt_name)

        logger.info(
            "[Self-Consistency] Generating %d samples...", self.num_samples
        )

        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []

        for i in range(self.num_samples):
            response = self.llm.generate(
                messages, temperature=self.temperature, top_p=self.top_p
            )
            all_responses.append(response)

            answers = parse_answer(response)
            for opt in answers:
                option_votes[opt] += 1

            logger.debug(
                "  Sample %d: %s", i + 1,
                sorted(answers) if answers else "No answer",
            )

        # 按阈值选项（对 D 采用更严格阈值，减少高风险误选）
        voted_answers = set()
        for opt, count in option_votes.items():
            threshold = (
                self.d_option_threshold if opt == "D" else self.vote_threshold
            )
            if count >= threshold:
                voted_answers.add(opt)

        # 处理边界情况：四个选项全部被选中
        if len(voted_answers) == 4:
            none_option = find_none_correct_option(item.options)
            if none_option and none_option in voted_answers:
                voted_answers.discard(none_option)
                logger.info(
                    "[Logic Check] Removed '%s' (conflicts with other selections)",
                    none_option,
                )
            else:
                vote_counts = [option_votes[opt] for opt in voted_answers]
                min_vote = min(vote_counts)
                max_vote = max(vote_counts)

                if min_vote <= 1 and max_vote >= 4:
                    weak_opts = [
                        opt for opt in voted_answers
                        if option_votes[opt] == min_vote
                    ]
                    if len(weak_opts) == 1:
                        voted_answers.discard(weak_opts[0])
                        logger.info(
                            "[Logic Check] Removed weak option '%s' "
                            "(votes: %d vs max: %d)",
                            weak_opts[0], min_vote, max_vote,
                        )

        vote_summary = ", ".join(
            f"{opt}:{count}" for opt, count in sorted(option_votes.items())
        )
        logger.info("[Vote counts] %s", vote_summary)

        # 若都不过阈值，回退为“最高票并列项”
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {
                    opt for opt, count in option_votes.items()
                    if count == max_votes
                }

        # 记录最终进入后处理前的选择集合
        logger.info(
            "[Threshold: general=%d, D=%d] Selected (after fallback): %s",
            self.vote_threshold,
            self.d_option_threshold,
            sorted(voted_answers),
        )

        # 记录不确定选项，便于后续调参分析
        uncertain_options = {
            opt for opt, count in option_votes.items()
            if 1 < count < self.vote_threshold
        }
        if uncertain_options:
            logger.info(
                "[Verification] Uncertain options: %s",
                sorted(uncertain_options),
            )

        final_answers = post_process_answers(voted_answers, item.options)

        output = (
            f"========== SELF-CONSISTENCY (Option-Level Voting) ==========\n"
            f"Samples: {self.num_samples}, Threshold: {self.vote_threshold}\n"
            f"Vote counts: {vote_summary}\n"
            f"Voted answers: {sorted(voted_answers)}\n"
            f"After post-processing: {sorted(final_answers)}\n\n"
            f"========== BEST RESPONSE ==========\n"
            f"{all_responses[0] if all_responses else 'No response'}\n\n"
            f"Final Answer I Reasoned: "
            f"{','.join(sorted(final_answers)) if final_answers else 'A'}"
        )
        return output


class ExperienceMemory:
    """持久化“经验样本”，用于跨会话检索复用。"""

    def __init__(self, memory_path: str = "data/memory/agent_memory.jsonl"):
        self.memory_path = memory_path
        self.max_records = 3000
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

    def retrieve(self, event: str, options: List[str], limit: int = 2) -> List[Dict[str, Any]]:
        """按词项 Jaccard 相似度检索最相关历史经验。"""
        if not os.path.exists(self.memory_path):
            return []

        query_tokens = self._tokenize(event + " " + " ".join(options))
        scored: List[Tuple[float, Dict[str, Any]]] = []

        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                memory_text = (
                    record.get("event", "")
                    + " "
                    + " ".join(record.get("options", []))
                    + " "
                    + " ".join(record.get("final_answers", []))
                )
                memory_tokens = self._tokenize(memory_text)
                score = self._jaccard(query_tokens, memory_tokens)
                if score > 0:
                    scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def append(self, record: Dict[str, Any]) -> None:
        """追加写入经验记录，并控制最大存储条数。"""
        lines: List[str] = []
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]

        lines.append(json.dumps(record, ensure_ascii=False))
        if len(lines) > self.max_records:
            lines = lines[-self.max_records:]

        with open(self.memory_path, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
            if token not in {"with", "that", "from", "this", "were", "have"}
        }

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0


class AgenticReActApproach(BaseApproach):
    """
    Agent 化策略，包含四个关键能力：
    1) 动态路由（Router）
    2) ReAct 迭代推理/检索
    3) Critic 反思纠错
    4) 持久化跨会话记忆
    """

    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.max_iterations = 3
        self.memory = ExperienceMemory()

    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        """
        核心执行流程：
        路由 -> 迭代 ReAct -> Critic 反思 -> 后处理 -> 写入记忆。
        """
        route = self._route_task(item)
        initial_documents = self._retrieve_documents(item)
        memory_examples = self.memory.retrieve(item.event, item.options, limit=2)

        thought_trace = []
        best_response = ""
        final_answers: Set[str] = set()
        current_documents = initial_documents

        for step in range(1, self.max_iterations + 1):
            docs_text, options_text = self._format_context(item, current_documents)
            react_response = self.llm.generate(
                self._build_react_messages(
                    item=item,
                    route=route,
                    docs_text=docs_text,
                    options_text=options_text,
                    memory_examples=memory_examples,
                    step=step,
                    previous_thoughts=thought_trace,
                ),
                temperature=0.2,
                top_p=0.95,
            )
            thought_trace.append(react_response)

            # 根据 Action 决定继续检索还是结束推理
            action = self._parse_action(react_response)
            candidate_answers = parse_answer(react_response)
            if candidate_answers:
                final_answers = candidate_answers
                best_response = react_response

            if action["type"] == "final":
                break
            if action["type"] == "search":
                current_documents = self._dynamic_retrieve(item, action["query"])
                continue
            if final_answers:
                break

        if not final_answers:
            final_answers = parse_answer(best_response)

        reflected_answers, critic_report = self._critic_reflect(
            item=item,
            route=route,
            thought_trace=thought_trace,
            candidate_answers=final_answers,
            documents=current_documents,
        )
        processed_answers = post_process_answers(reflected_answers, item.options)

        output = self._build_agent_output(
            route=route,
            thought_trace=thought_trace,
            critic_report=critic_report,
            final_answers=processed_answers,
        )

        self.memory.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event_id": item.event_id,
                "event": item.event,
                "options": item.options,
                "route": route,
                "final_answers": sorted(processed_answers),
                "critic_report": critic_report[:500],
            }
        )
        return output

    def _route_task(self, item: AERItem) -> str:
        """Router：将任务划分为信息缺失/逻辑冲突/多跳因果三类。"""
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a router agent. Classify question type as exactly one of: "
                    "INFO_GAP, LOGICAL_CONFLICT, MULTI_HOP_CAUSAL."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Event:\n{item.event}\n\nOptions:\n{options_text}\n\n"
                    "Respond with one label only."
                ),
            },
        ]
        response = (self.llm.generate(messages, temperature=0, top_p=1) or "").upper()
        for label in ["INFO_GAP", "LOGICAL_CONFLICT", "MULTI_HOP_CAUSAL"]:
            if label in response:
                return label
        return "MULTI_HOP_CAUSAL"

    def _build_react_messages(
        self,
        item: AERItem,
        route: str,
        docs_text: str,
        options_text: str,
        memory_examples: List[Dict[str, Any]],
        step: int,
        previous_thoughts: List[str],
    ) -> List[Dict[str, str]]:
        """构造 ReAct 单轮消息，包含历史思考和记忆示例。"""
        memory_block = "No prior memory examples."
        if memory_examples:
            rendered = []
            for idx, mem in enumerate(memory_examples, 1):
                rendered.append(
                    f"[Memory {idx}] Event: {mem.get('event', '')}\n"
                    f"Route: {mem.get('route', '')}\n"
                    f"Final: {','.join(mem.get('final_answers', [])) or 'N/A'}"
                )
            memory_block = "\n\n".join(rendered)

        prior = "\n\n".join(previous_thoughts[-2:]) if previous_thoughts else "N/A"
        user_prompt = (
            f"Step: {step}/{self.max_iterations}\n"
            f"Route: {route}\n\n"
            f"Target Event:\n{item.event}\n\n"
            f"Evidence Documents:\n{docs_text}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Similar Historical Memory:\n{memory_block}\n\n"
            f"Recent Thinking:\n{prior}\n\n"
            "Use ReAct format:\n"
            "Thought: ...\n"
            "Action: SEARCH[query] or FINAL[labels]\n"
            "Observation: ...\n"
            "Final Answer I Reasoned: A or A,B (only if Action is FINAL)\n"
            "Rules: conservative selection; never output empty final answer."
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are an autonomous abductive reasoning agent. "
                    "Think, decide the next action, and iterate with evidence."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    def _parse_action(self, response: str) -> Dict[str, str]:
        """解析模型动作：SEARCH[...] / FINAL[...]。"""
        search_match = re.search(r"Action:\s*SEARCH\[(.*?)\]", response, re.IGNORECASE)
        if search_match:
            return {"type": "search", "query": search_match.group(1).strip()}
        final_match = re.search(r"Action:\s*FINAL\[(.*?)\]", response, re.IGNORECASE)
        if final_match:
            return {"type": "final", "query": final_match.group(1).strip()}
        if parse_answer(response):
            return {"type": "final", "query": ""}
        return {"type": "none", "query": ""}

    def _dynamic_retrieve(self, item: AERItem, query: str) -> List[str]:
        """根据 ReAct 生成的新查询执行动态检索。"""
        if not self.retriever:
            return item.documents
        return self.retriever.retrieve(
            event=query or item.event,
            title_snippet=item.title_snippet,
            documents=item.documents,
            options=item.options,
        )

    def _critic_reflect(
        self,
        item: AERItem,
        route: str,
        thought_trace: List[str],
        candidate_answers: Set[str],
        documents: List[str],
    ) -> Tuple[Set[str], str]:
        """Critic 节点：审查候选答案并做保守纠偏。"""
        docs_text, options_text = self._format_context(item, documents)
        trace_text = "\n\n".join(thought_trace[-2:]) if thought_trace else "N/A"
        candidate_text = ",".join(sorted(candidate_answers)) if candidate_answers else "A"

        critic_messages = [
            {
                "role": "system",
                "content": (
                    "You are a critic agent for causal consistency checking. "
                    "Challenge weak links, temporal conflicts, and unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Route: {route}\n"
                    f"Target Event:\n{item.event}\n\n"
                    f"Options:\n{options_text}\n\n"
                    f"Documents:\n{docs_text}\n\n"
                    f"Candidate answer: {candidate_text}\n\n"
                    f"Reasoning trace:\n{trace_text}\n\n"
                    "Task:\n"
                    "1) Evaluate whether each selected option is directly supported.\n"
                    "2) If any selected option lacks support, remove it.\n"
                    "3) If all are unsupported, choose the least risky single option.\n"
                    "Output last line as: Final Answer I Reasoned: X or X,Y"
                ),
            },
        ]
        critic_response = self.llm.generate(critic_messages, temperature=0, top_p=1)
        critic_answers = parse_answer(critic_response)
        if not critic_answers:
            critic_answers = candidate_answers if candidate_answers else {"A"}
        return critic_answers, critic_response

    def _build_agent_output(
        self,
        route: str,
        thought_trace: List[str],
        critic_report: str,
        final_answers: Set[str],
    ) -> str:
        """拼接完整可审计输出（轨迹 + critic + final）。"""
        trace_preview = "\n\n".join(
            f"--- Iteration {idx + 1} ---\n{trace}"
            for idx, trace in enumerate(thought_trace)
        )
        return (
            "========== AGENTIC REACT ORCHESTRATION ==========\n"
            f"Route Decision: {route}\n"
            f"Iterations Used: {len(thought_trace)}\n\n"
            "========== REACT TRACE ==========\n"
            f"{trace_preview}\n\n"
            "========== CRITIC REPORT ==========\n"
            f"{critic_report}\n\n"
            "========== FINAL ==========\n"
            f"Final Answer I Reasoned: {','.join(sorted(final_answers)) if final_answers else 'A'}"
        )
