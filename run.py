import argparse
import concurrent.futures
from datetime import datetime
import os
import json
import time
import logging
import threading

from src.dataloader import DataLoader
from src.llm import ChatLLM
from src.approaches import (
    BaselineApproach,
    SelfConsistencyRefinementApproach,
    ConservativeApproach,
    LightweightConsistencyApproach,
    TwoPassApproach,
    AgenticReActApproach,
    parse_answer,
)
from src.evaluator import Evaluator
from src.retriever import DocumentRetriever
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 所有可选推理策略注册表：
# 通过 --approach 参数选择对应类并实例化。
APPROACHES = {
    "baseline": BaselineApproach,
    "sc_refine": SelfConsistencyRefinementApproach,
    "conservative": ConservativeApproach,
    "lightweight_sc": LightweightConsistencyApproach,
    "twopass_real": TwoPassApproach,
    "agentic_react": AgenticReActApproach,
}
# 所有可选 Prompt 模板名称（在 src/prompts.py 中定义）
PROMPT_NAMES = ["cot", "conservative", "evidence_anchored", "balanced"]


def parse_ground_truth(answer_str: str) -> set:
    """将标注答案字符串（如 'A,C'）解析为选项集合。"""
    if not answer_str:
        return set()
    answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
    return {a for a in answers if a in ["A", "B", "C", "D"]}


def main():
    """程序主入口：配置 -> 加载 -> 并发推理 -> 评测 -> 落盘。"""
    parser = argparse.ArgumentParser(
        description="SemEval 2026 Task 12: Abductive Event Reasoning"
    )
    # =========================
    # 数据与输出路径参数
    # =========================
    parser.add_argument("--docs_path", type=str, default="data/dev/docs.json")
    parser.add_argument(
        "--questions_path", type=str, default="data/dev/questions.jsonl"
    )
    parser.add_argument("--submission_path", type=str, default="submission.jsonl")
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results",
    )

    # =========================
    # 检索相关参数
    # =========================
    parser.add_argument(
        "--no_retrieval", action="store_true",
        help="Disable document retrieval (use all documents)",
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of top documents to retrieve (0 = use all)",
    )
    parser.add_argument(
        "--use_full_content", action="store_true",
        help="Use full document content for retrieval",
    )
    parser.add_argument(
        "--use_gpu", action="store_true",
        help="Use GPU for semantic retrieval (if available)",
    )
    parser.add_argument(
        "--use_per_option", action="store_true",
        help="Use per-option retrieval (retrieve for event + each option)",
    )

    # =========================
    # 推理策略与 Prompt 参数
    # =========================
    parser.add_argument(
        "--approach", type=str, default="baseline",
        choices=list(APPROACHES.keys()),
        help="Reasoning approach to use",
    )
    parser.add_argument(
        "--prompt_name", type=str, default="cot",
        choices=PROMPT_NAMES,
        help="Prompt template to use",
    )
    args = parser.parse_args()

    load_dotenv()

    # 从环境变量读取模型配置
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    max_workers = int(os.getenv("MAX_WORKERS", "4"))

    if not all([model_name, api_key, base_url]):
        logger.error(
            "Missing required env vars. Please set MODEL_NAME, API_KEY, "
            "BASE_URL in .env file."
        )
        return

    # 初始化 LLM 客户端
    llm = ChatLLM(model_name=model_name, api_key=api_key, base_url=base_url)

    # 初始化检索器：可关闭检索，直接使用全量文档
    retriever = (
        None
        if args.no_retrieval
        else DocumentRetriever(
            top_k=args.top_k if args.top_k > 0 else 10,
            use_full_content=args.use_full_content,
            use_gpu=args.use_gpu,
            use_per_option=args.use_per_option,
        )
    )

    # 动态装配策略类
    ApproachClass = APPROACHES[args.approach]
    solver = ApproachClass(llm, retriever)
    # 初始化数据加载器与评测器
    loader = DataLoader(args.docs_path, args.questions_path)
    evaluator = Evaluator()
    # 多线程写入共享对象时使用锁，避免竞态条件
    evaluator_lock = threading.Lock()
    submission = []
    submission_lock = threading.Lock()
    start_time = time.time()

    # 尝试估算总题量，供进度条显示
    try:
        with open(args.questions_path, "r", encoding="utf-8") as f:
            total_questions = sum(1 for _ in f)
    except Exception:
        total_questions = None

    logger.info("Running experiment with approach=%s", args.approach)
    logger.info("Using prompt_name=%s", args.prompt_name)
    if retriever is not None:
        logger.info("Document retrieval: Enabled (top_k=%d)", args.top_k)
    else:
        logger.info("Document retrieval: Disabled (using all documents)")

    is_test_split = args.questions_path.endswith("test/questions.jsonl")

    # 并发执行每条样本，提升评测吞吐
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        events = list(loader.load())
        future_to_event = {
            executor.submit(solver.solve, event, prompt_name=args.prompt_name): event
            for event in events
        }

        pbar = tqdm(
            total=total_questions if total_questions is not None else len(events),
            desc="Progress",
            ncols=80,
        )

        for future in concurrent.futures.as_completed(future_to_event):
            event = future_to_event[future]
            try:
                # 1) 模型推理
                prediction = future.result()

                # 2) 解析预测与标注
                predicted = parse_answer(prediction)
                ground_truth = parse_ground_truth(event.answer)

                # 3) 更新评测统计（加锁）
                with evaluator_lock:
                    evaluator.update(
                        predicted=predicted,
                        ground_truth=ground_truth,
                        event_id=event.event_id,
                        prediction_text=prediction,
                        event=event.event,
                        options=event.options,
                    )

                # 4) 生成提交格式（按官方要求输出选项字符串）
                predicted_str = ",".join(sorted(predicted))
                with submission_lock:
                    submission.append(
                        {"id": event.event_id, "answer": predicted_str}
                    )

            except Exception as e:
                # 若某条样本异常，记录日志并按空预测计入评测
                logger.error(
                    "Event %s generated an exception: %s", event.event_id, e
                )
                with evaluator_lock:
                    evaluator.update(
                        predicted=set(),
                        ground_truth=parse_ground_truth(event.answer),
                        event_id=event.event_id,
                        prediction_text="",
                        event=event.event,
                        options=event.options,
                    )
            finally:
                pbar.update(1)
        pbar.close()

    end_time = time.time()
    total_time = end_time - start_time

    # 控制台打印评测摘要
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"LLM Model: {model_name}")
    print(f"Approach: {args.approach}")
    print(f"Prompt Type: {args.prompt_name}")
    print(f"Retrieval: {not args.no_retrieval}")
    if not args.no_retrieval:
        print(f"  Top-K: {args.top_k}")
        print(f"  Full Content: {args.use_full_content}")
        print(f"  Per-Option: {args.use_per_option}")
    print(f"Total Time: {total_time:.2f} seconds")

    if not is_test_split:
        summary = evaluator.get_summary()
        print(f"\nTotal: {summary['total']}")
        print(f"Full Match: {summary['full_match']}")
        print(f"Partial Match: {summary['partial_match']}")
        print(f"Incorrect: {summary['incorrect']}")
        print(f"Official Score: {summary['official_score']:.4f}")
        print(
            f"Strict Accuracy: {summary['strict_accuracy']:.4f} "
            f"({summary['strict_accuracy'] * 100:.2f}%)"
        )
        print(f"Macro F1 Score: {summary['macro_f1']:.4f}")
        print(
            f"\nSingle Answer Accuracy: "
            f"{summary['single_answer_accuracy']:.4f} "
            f"({summary['single_answer_count']} cases)"
        )
        print(
            f"Multi Answer Accuracy: "
            f"{summary['multi_answer_accuracy']:.4f} "
            f"({summary['multi_answer_count']} cases)"
        )
        print(
            f"Insufficient Info Accuracy: "
            f"{summary['insufficient_info_accuracy']:.4f} "
            f"({summary['insufficient_info_count']} cases)"
        )
        print("\nOption Level Matrix:")
        print("\tPrecision\tRecall\t\tF1")
        for option, matrix in sorted(summary["option_matrix"].items()):
            print(
                f"{option}\t{matrix['precision']:.4f}\t\t"
                f"{matrix['recall']:.4f}\t\t{matrix['f1']:.4f}"
            )
    print("=" * 50)

    # 保存详细评测结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    evaluator.save_results(output_file, approach_name=solver.__class__.__name__)

    # 保存提交文件
    with open(args.submission_path, "w", encoding="utf-8") as f:
        for item in submission:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Evaluation complete! Submission saved to %s, results to %s",
        args.submission_path, output_file,
    )


if __name__ == "__main__":
    # 避免 tokenizer 并行导致的告警/资源争抢
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
