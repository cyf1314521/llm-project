import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AERItem:
    """单条 AER 样本的数据结构。"""
    topic_id: int
    event: str
    event_id: str
    title_snippet: List[str]
    documents: List[str]
    options: List[str]
    answer: Optional[str]


class DataLoader:
    """加载文档与问题，并按 topic_id 进行关联。"""

    def __init__(self, docs_path: str, questions_path: str):
        self.docs_path = docs_path
        self.questions_path = questions_path
        # 预加载文档 JSON，避免每条问题重复读取
        self.docs_data = self._load_json_data()

    def _load_json_data(self):
        """读取 docs.json。读取失败时返回空列表并记录日志。"""
        try:
            # utf-8-sig 可自动去掉 Windows 常见 BOM 头
            with open(self.docs_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("JSON file not found at %s", self.docs_path)
            return []
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from %s", self.docs_path)
            return []

    def load(self):
        """
        逐条产出 AERItem：
        1) 先将文档数据整理为 topic_id -> docs/title_snippet
        2) 再逐行读取 questions.jsonl 并拼接对应文档
        """
        docs_dict = {}
        title_snippet_dict = {}
        for item in self.docs_data:
            topic_id = item["topic_id"]
            # full content 文档列表
            docs_dict[topic_id] = [
                doc.get("content", "") for doc in item["docs"]
            ]
            # title + snippet 文档列表（供轻量检索模式）
            title_snippet_dict[topic_id] = [
                doc.get("title", "") + " " + doc.get("snippet", "")
                for doc in item["docs"]
            ]

        with open(self.questions_path, "r", encoding="utf-8-sig") as f:
            for line_str in f:
                line_str = line_str.strip()
                if not line_str:
                    continue
                try:
                    line = json.loads(line_str)
                except json.JSONDecodeError:
                    # 问题文件出现坏行时不中断全流程
                    logger.warning(
                        "Skipping invalid JSONL line (first 120 chars): %r",
                        line_str[:120],
                    )
                    continue

                topic_id = line["topic_id"]
                # 通过 yield 形成流式加载，节省内存
                yield AERItem(
                    topic_id=topic_id,
                    event=line["target_event"],
                    event_id=line["id"],
                    title_snippet=title_snippet_dict.get(topic_id, []),
                    documents=docs_dict.get(topic_id, []),
                    options=[line[f"option_{i}"] for i in ["A", "B", "C", "D"]],
                    answer=line.get("golden_answer"),
                )
