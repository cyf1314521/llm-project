import argparse
import concurrent.futures
from datetime import datetime
import os
import re
import json
import time
from src.dataloader import DataLoader
from src.llm import ChatLLM
from src.approaches import (
    BaselineApproach,
    SelfConsistencyRefinementApproach,
    ConservativeApproach,
    LightweightConsistencyApproach,
    TwoPassApproach,
)
from src.evaluator import Evaluator
from src.retriever import DocumentRetriever
from dotenv import load_dotenv
from tqdm import tqdm

# Approach registry for CLI selection
APPROACHES = {
    "baseline": BaselineApproach,
    "sc_refine": SelfConsistencyRefinementApproach,
    "conservative": ConservativeApproach,
    "lightweight_sc": LightweightConsistencyApproach,
    "twopass_real": TwoPassApproach,
}
PROMPT_NAME = [
    "cot",
    "conservative",
    "evidence_anchored",
    "balanced",
]


def parse_answer(prediction: str) -> set:
    if not prediction:
        return set()

    # Try to find "Final Answer I Reasoned: ..." pattern
    pattern = r"Final Answer I Reasoned:\s*([A-D](?:\s*,\s*[A-D])*)"
    
    matches = re.findall(pattern, prediction, re.IGNORECASE)
    
    if matches:
        answer_str = matches[-1].strip()
        answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
        valid_answers = {a for a in answers if a in ["A", "B", "C", "D"]}
        return valid_answers

    # Fallback: try to find any single letter A-D at the end
    pattern2 = r"\b([A-D])\b"
    matches = re.findall(pattern2, prediction[-200:])
    if matches:
        return {m.upper() for m in matches if m.upper() in ["A", "B", "C", "D"]}

    return set()

def parse_ground_truth(answer_str: str) -> set:
    if not answer_str:
        return set()
    answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
    return {a for a in answers if a in ["A", "B", "C", "D"]}


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", type=str, default="data/dev/docs.json")
    parser.add_argument(
        "--questions_path", type=str, default="data/dev/questions.jsonl"
    )
    parser.add_argument("--submission_path", type=str, default="submission.jsonl")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )

    # arguments for retrieval
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="Disable document retrieval (use all documents)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top documents to retrieve (0 = use all)",
    )
    parser.add_argument(
        "--use_full_content",
        action="store_true",
        help="Use full document content for retrieval",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for semantic retrieval (if available)",
    )
    parser.add_argument(
        "--use_per_option",
        action="store_true",
        help="Use per-option retrieval (retrieve for event + each option)",
    )

    # arguments for approach
    parser.add_argument(
        "--approach",
        type=str,
        default="baseline",
        choices=list(APPROACHES.keys()),
        help="Reasoning approach to use (baseline, sc_refine)",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="cot",
        choices=PROMPT_NAME,
        help="Prompt to use (cot, optimized, twopass, structured)",
    )
    args = parser.parse_args()

    # initialize components
    load_dotenv()
    llm = ChatLLM(
        model_name=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

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

    # Select approach based on CLI argument
    ApproachClass = APPROACHES[args.approach]
    solver = ApproachClass(llm, retriever)
    loader = DataLoader(args.docs_path, args.questions_path)
    evaluator = Evaluator()
    submission = []
    start_time = time.time()

    # count questions.jsonl
    try:
        with open(args.questions_path, "r", encoding="utf-8") as f:
            total_questions = sum(1 for _ in f)
    except Exception as e:
        total_questions = None

    print(f"Running experiment with {args.approach}...")
    print(f"Using {args.prompt_name} as prompt...")
    if retriever != None:
        print(f"Document retrieval: Enabled (top_k={args.top_k})\n")
    else:
        print("Document retrieval: Disabled (using all documents)\n")
    print()

    max_workers = int(os.getenv("MAX_WORKERS"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        events = list(loader.load())
        future_to_event = {
            executor.submit(solver.solve, event, prompt_name=args.prompt_name): event for event in events
        }

        pbar = tqdm(total=total_questions if total_questions is not None else len(events), desc="Progress", ncols=80)

        for future in concurrent.futures.as_completed(future_to_event):
            event = future_to_event[future]
            try:
                prediction = future.result()

                # parse answers
                predicted = parse_answer(prediction)
                ground_truth = parse_ground_truth(event.answer)

                # update evaluator
                evaluator.update(
                    predicted=predicted,
                    ground_truth=ground_truth,
                    event_id=event.event_id,
                    prediction_text=prediction,
                    event=event.event,
                    options=event.options,
                )

                # save to submission
                predicted_str = ",".join(sorted(predicted))
                submission_answer = {"id": event.event_id, "answer": predicted_str}
                submission.append(submission_answer)

            except Exception as e:
                print(f"{event.id} generated an exception: {e}")
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


    # print evaluation summary
    print("\n")
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"LLM Model: {os.getenv('MODEL_NAME')}")
    print(f"Approach: {args.approach}")
    print(f"Prompt Type: {args.prompt_name}")
    print(f"Retrieval: {not args.no_retrieval}")
    if not args.no_retrieval:
        print(f"Use Full Content: {args.use_full_content}")
    print(f"Total Time: {total_time:.2f} seconds")
    if 'test' not in args.docs_path:
        summary = evaluator.get_summary()
        print(f"\nTotal: {summary['total']}")
        print(f"Full Match: {summary['full_match']}")
        print(f"Partial Match: {summary['partial_match']}")
        print(f"Incorrect: {summary['incorrect']}")
        print(f"Official Score: {summary['official_score']:.4f}")
        print(f"Strict Accuracy: {summary['strict_accuracy']:.4f} ({summary['strict_accuracy'] * 100:.2f}%)")
        print(f"Macro F1 Score: {summary['macro_f1']:.4f}")
        print(
            f"\nSingle Answer Accuracy: {summary['single_answer_accuracy']:.4f} ({summary['single_answer_count']} cases)"
        )
        print(
            f"Multi Answer Accuracy: {summary['multi_answer_accuracy']:.4f} ({summary['multi_answer_count']} cases)"
        )
        print(
            f"Insufficient Info Accuracy: {summary['insufficient_info_accuracy']:.4f} ({summary['insufficient_info_count']} cases)"
        )
        print(f"\nOption Level Matrix:")
        print("\tPrecision\tRecall\t\tF1")
        for option, matrix in sorted(summary["option_matrix"].items()):
            print(
                f"{option}\t{matrix['precision']:.4f}\t\t{matrix['recall']:.4f}\t\t{matrix['f1']:.4f}"
            )
    print("=" * 50)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    evaluator.save_results(output_file, approach_name=solver.__class__.__name__)

    # write to submission.jsonl
    with open(args.submission_path, "w", encoding="utf-8") as f:
        for item in submission:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nEvaluation complete! submission.json created and results saved to: {output_file}\n")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
