import json
from dataclasses import dataclass
from typing import List

@dataclass
class AERItem:
    """
    A customized data type class.
    """
    id: int
    event: str
    event_id: str
    title_snippet: List[str]
    documents: List[str]
    options: List[str]
    answer: str

class DataLoader:
    """
    The main class for data loader.
    """
    def __init__(self, docs_path: str, questions_path: str):
        self.docs_path = docs_path
        self.questions_path = questions_path

        # We load docs.json file in advance, cuz it would be used every iteration.
        self.docs_data = self._load_json_data()

    def _load_json_data(self):
        try:
            with open(self.docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.docs_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {self.docs_path}")
            return []

    def load(self):
        # reformat the docs_data into a dict like {<topic_id>: [<corresponding docs content>, ...]}
        docs_dict = {}
        title_snippet_dict = {}
        for item in self.docs_data:
            docs_dict[item["topic_id"]] = [doc.get("content", "") for doc in item["docs"]]
            # Splice title and snipprt of the document together, forming this title_snippet
            title_snippet_dict[item["topic_id"]] = [doc.get("title", "") + " " + doc.get("snippet", "") for doc in item["docs"]]

        with open(self.questions_path, "r", encoding="utf-8") as f:
            for line_str in f:
                try:
                    line = json.loads(line_str)
                except json.JSONDecodeError:
                    continue
                topic_id = line["topic_id"]
                documents = docs_dict.get(topic_id, [])
                title_snippet = title_snippet_dict.get(topic_id, [])
                aer_item = AERItem(
                    id = topic_id,
                    event = line["target_event"],
                    event_id = line["id"],
                    title_snippet = title_snippet,
                    documents = documents,
                    options = [line[f"option_{i}"] for i in ["A", "B", "C", "D"]],
                    answer = line['golden_answer'] if 'golden_answer' in line else None
                )
                yield aer_item