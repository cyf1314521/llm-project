"""
Approaches for SemEval 2026 Task 12: Abductive Event Reasoning

Optimized for evaluation metric:
- 1.0 points: Perfect match
- 0.5 points: Partial match (subset, no wrong)
- 0.0 points: Any wrong selection

Key insight: Conservative strategy is optimal.
"""

from abc import ABC, abstractmethod
from itertools import count

from torch import threshold
from src.llm import BaseLLM
from src.retriever import DocumentRetriever
from src.dataloader import AERItem
from collections import Counter
import re
from src.prompts import PROMPTS


class BaseApproach(ABC):
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def solve(self, item: AERItem, prompt_name: str) -> str:
        pass
    
    def _parse_answer_from_response(self, response: str) -> set:
        """Extract answer options from LLM response."""
        if not response:
            return set()
        
        # Try to find "Final Answer I Reasoned: ..." pattern
        pattern = r"Final Answer I Reasoned:\s*([A-D](?:\s*,\s*[A-D])*)"
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            answer_str = match.group(1).strip()
            answers = [a.strip().upper() for a in answer_str.split(",") if a.strip()]
            return {a for a in answers if a in ["A", "B", "C", "D"]}
        
        # Fallback: find any A-D letters in last 200 chars
        pattern2 = r"\b([A-D])\b"
        matches = re.findall(pattern2, response[-200:])
        if matches:
            return {m.upper() for m in matches if m.upper() in ["A", "B", "C", "D"]}
        
        return set()


# ============================================================
# Post-processing utility functions
# ============================================================

def detect_duplicate_options(options: list) -> list:
    """
    Detect duplicate or nearly identical options
    
    Returns:
        list of tuples: [(idx1, idx2, "identical"), ...]
    """
    labels = ["A", "B", "C", "D"]
    duplicates = []
    
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            # Standardize comparison: strip and lowercase
            opt_i = options[i].strip().lower()
            opt_j = options[j].strip().lower()
            
            if opt_i == opt_j:
                duplicates.append((labels[i], labels[j], "identical"))
            # More similarity detection logic can be added here
    
    return duplicates


def find_none_correct_option(options: list) -> str:
    """
    Find options like "None of the others are correct"
    
    Returns:
        Option label (A/B/C/D) or None
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
    Post-process answers, enforce logical rules
    
    Rules:
    1. Duplicate options must be both selected or both not selected
    2. "None correct" cannot be selected with other options
    3. Answers cannot be empty
    """
    if not answers:
        return answers
    
    processed = answers.copy()
    
    # Rule 1: Handle duplicate options
    duplicates = detect_duplicate_options(options)
    for label1, label2, dup_type in duplicates:
        # If one is selected, both must be selected
        if label1 in processed or label2 in processed:
            processed.add(label1)
            processed.add(label2)
    
    # Rule 2: Handle mutual exclusivity of "None correct"
    none_label = find_none_correct_option(options)
    if none_label and none_label in processed:
        # If "None correct" is selected, check if other options are also selected
        other_answers = processed - {none_label}
        if other_answers:
            # Conflict! According to conservative strategy, remove "None correct" (keep substantive answers)
            processed.discard(none_label)
    
    return processed


# ============================================================
# Conservative Approach
# ============================================================

class ConservativeApproach(BaseApproach):
    """
    Conservative approach optimized for partial matching metric.
    
    Key principle: Better to miss correct answers than select wrong ones.
    - Wrong selection = 0 points
    - Partial correct = 0.5 points
    """
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # Retrieve relevant documents
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        
        # Get prompt
        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Generate response
        response = self.llm.generate(messages)
        
        # Parse answer
        raw_answers = self._parse_answer_from_response(response)
        
        # Post-processing
        processed_answers = post_process_answers(raw_answers, item.options)
        
        # If post-processing changed the answer, append explanation
        if processed_answers != raw_answers:
            response += f"\n\n[Post-processing applied: {raw_answers} -> {processed_answers}]"
        
        return response


# ============================================================
# Lightweight Consistency Approach
# ============================================================

class LightweightConsistencyApproach(BaseApproach):
    """
    Lightweight Self-Consistency with option-level voting.
    
    Only 3 samples (not 5+), no refinement step.
    Uses option-level voting instead of answer-set voting.
    """
    
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 3
        self.temperature = 0.5
        self.vote_threshold = 2  # At least 2/3 to select
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # Retrieve relevant documents
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        
        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )
        
        # ============ Multiple sampling ============
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []
        
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=self.temperature)
            all_responses.append(response)
            
            # Parse answer and vote
            answers = self._parse_answer_from_response(response)
            for opt in answers:
                option_votes[opt] += 1
        
        # ============ Option-level voting ============
        # Only select options above threshold
        voted_answers = {opt for opt, count in option_votes.items() 
                        if count >= self.vote_threshold}
        
        # If no option passes threshold, select those with highest votes
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {opt for opt, count in option_votes.items() 
                               if count == max_votes}
        
        # ============ Post-processing ============
        final_answers = post_process_answers(voted_answers, item.options)
        
        # Build output
        vote_summary = ", ".join(f"{opt}:{count}" for opt, count in sorted(option_votes.items()))
        
        output = f"""
        ========== LIGHTWEIGHT CONSISTENCY ==========
        Samples: {self.num_samples}, Threshold: {self.vote_threshold}
        Vote counts: {vote_summary}
        Voted answers: {sorted(final_answers)}

        ========== BEST RESPONSE ==========
        {all_responses[0] if all_responses else "No response"}

        Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
        """
        return output


# ============================================================
# Two-Pass Approach (True two-call)
# ============================================================

class TwoPassApproach(BaseApproach):
    """
    True two-pass approach with separate API calls.
    
    Pass 1: Liberal candidate selection (high recall)
    Pass 2: Strict causal verification (high precision)
    """
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        # Retrieve relevant documents
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )
        
        # ============ PASS 1: Liberal selection ============
        pass1_system = "You are an expert in causal reasoning. Your task is to identify ALL potentially relevant options."
        pass1_user = f"""
        TARGET EVENT: {item.event}

        DOCUMENTS:
        {docs_text}

        OPTIONS:
        {options_text}

        TASK: For each option, determine if it has ANY potential connection to the target event.
        Be INCLUSIVE at this stage - mark as CANDIDATE if there's any possible relationship.

        For each option, answer:
        - Option A: CANDIDATE or REJECT? (one word)
        - Option B: CANDIDATE or REJECT? (one word)
        - Option C: CANDIDATE or REJECT? (one word)
        - Option D: CANDIDATE or REJECT? (one word)

        Then list all CANDIDATE options.
        """
        
        pass1_response = self.llm.generate([
            {"role": "system", "content": pass1_system},
            {"role": "user", "content": pass1_user}
        ], temperature=0.3, top_p=0.9)
        
        # Parse Pass 1 candidates
        candidates = set()
        for label in ["A", "B", "C", "D"]:
            # Check if this option is marked as CANDIDATE
            if re.search(rf"Option {label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE):
                candidates.add(label)
            elif re.search(rf"{label}[:\s]*CANDIDATE", pass1_response, re.IGNORECASE):
                candidates.add(label)
        
        # If no clear candidates found, try other parsing methods
        if not candidates:
            match = re.search(r"candidates?[:\s]*([A-D](?:\s*,\s*[A-D])*)", pass1_response, re.IGNORECASE)
            if match:
                candidates = {c.strip().upper() for c in match.group(1).split(",") if c.strip().upper() in ["A", "B", "C", "D"]}
        
        # If still none, default all options as candidates
        if not candidates:
            candidates = {"A", "B", "C", "D"}
        
        # ============ PASS 2: Strict verification ============
        candidates_text = ", ".join(sorted(candidates))
        pass2_system = """
        You are an expert in causal reasoning. Your task is to verify which candidates are TRUE CAUSES.

        CRITICAL SCORING RULE:
        - Selecting ANY wrong option = 0 points
        - Missing some correct options = 0.5 points
        - Be CONSERVATIVE: Only select options you are CERTAIN about.
        """
        
        pass2_user = f"""
        TARGET EVENT: {item.event}

        DOCUMENTS:
        {docs_text}

        CANDIDATE OPTIONS (from Pass 1): {candidates_text}

        For each candidate, verify:
        1. TEMPORAL: Does evidence show this happened BEFORE the target event? (YES/NO)
        2. CAUSAL: Is there a clear mechanism by which this CAUSED the event? (YES/NO)
        3. EVIDENCE: Is there direct documentary support? (YES/NO)

        Only select options with ALL THREE = YES.

        Remember: Wrong selection = 0 points. Be conservative!

        Final Answer I Reasoned: [Only verified options]
        """
        
        pass2_response = self.llm.generate([
            {"role": "system", "content": pass2_system},
            {"role": "user", "content": pass2_user}
        ], temperature=0.1, top_p=1)
        
        # Parse final answer
        raw_answers = self._parse_answer_from_response(pass2_response)
        
        # Post-processing
        final_answers = post_process_answers(raw_answers, item.options)
        
        # Build output
        output = f"""
        ========== TWO-PASS APPROACH ==========

        ----- PASS 1: Candidate Selection -----
        Candidates identified: {sorted(candidates)}

        {pass1_response}

        ----- PASS 2: Strict Verification -----
        {pass2_response}

        ----- POST-PROCESSING -----
        Raw answers: {sorted(raw_answers)}
        Final answers: {sorted(final_answers)}

        Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
        """
        return output


# ============================================================
# Retain original Approaches (backward compatibility)
# ============================================================

class BaselineApproach(BaseApproach):
    """
    The basic zero-shot CoT approach.
    """

    def solve(self, item: AERItem, prompt_name: str = "cot") -> str:
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )

        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )

        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.generate(messages)
        
        # New: post-processing
        raw_answers = self._parse_answer_from_response(response)
        processed_answers = post_process_answers(raw_answers, item.options)
        
        if processed_answers != raw_answers:
            response += f"\n\n[Post-processing: {raw_answers} -> {processed_answers}]"
        
        return response


class SelfConsistencyRefinementApproach(BaseApproach):
    """
    Combines Self-Consistency (multiple sampling + voting) with Self-Refinement.
    
    UPDATED: Now uses option-level voting instead of answer-set voting.
    """
    
    def __init__(self, llm: BaseLLM, retriever: DocumentRetriever = None):
        super().__init__(llm, retriever)
        self.num_samples = 7
        self.temperature = 0.5
        self.top_p = 0.95
        self.vote_threshold = 4  # At least 4/7 to select
        self.d_option_threshold = 5  # Stricter for option D
    
    def _get_prompt(self, item: AERItem, prompt_name: str) -> tuple:
        """Get the system and user prompts."""
        documents = (
            self.retriever.retrieve(item.event, item.title_snippet, item.documents, item.options)
            if self.retriever
            else item.documents
        )
        
        docs_text = "\n".join(f"[Doc{i + 1}]: {doc}" for i, doc in enumerate(documents))
        options_text = "\n".join(
            f"{label}: {opt}" for label, opt in zip(["A", "B", "C", "D"], item.options)
        )

        system_prompt = PROMPTS[prompt_name]["system_prompt"]
        user_prompt = PROMPTS[prompt_name]["user_prompt"].format(
            event=item.event,
            docs_text=docs_text,
            options_text=options_text
        )
        
        return system_prompt, user_prompt, docs_text, options_text, item.event
    
    def solve(self, item: AERItem, prompt_name: str = "conservative") -> str:
        """
        Main solving method with improved option-level voting.
        """
        system_prompt, user_prompt, docs_text, options_text, event = self._get_prompt(item, prompt_name)
        
        # ============ STAGE 1: Option-level voting ============
        print(f"\n[Self-Consistency] Generating {self.num_samples} samples...")
        
        option_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        all_responses = []
        
        for i in range(self.num_samples):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.llm.generate(messages, temperature=self.temperature, top_p=self.top_p)
            all_responses.append(response)
            
            # Option-level voting
            answers = self._parse_answer_from_response(response)
            for opt in answers:
                option_votes[opt] += 1
            
            print(f"  Sample {i+1}: {sorted(answers) if answers else 'No answer'}")
        
        # Select based on threshold
        # Voting logic changed to
        voted_answers = set()
        for opt, count in option_votes.items():
            threshold = self.d_option_threshold if opt == 'D' else self.vote_threshold
            if count >= threshold:
                voted_answers.add(opt)
        
        # Improved 4-option restriction logic: only remove in clear anomalies
        if len(voted_answers) == 4:
            # Check if "None of the others" option is included
            none_option = find_none_correct_option(item.options)
            if none_option and none_option in voted_answers:
                # If all 4 are selected and "None" is included, this is a conflict, remove "None"
                voted_answers.discard(none_option)
                print(f"[Logic Check] Removed '{none_option}' (conflicts with other selections)")
            else:
                # Check for clear weak option (votes much lower than others)
                vote_counts = [option_votes[opt] for opt in voted_answers]
                min_vote = min(vote_counts)
                max_vote = max(vote_counts)
                
                # Only remove if weakest option votes <= 1 and strongest >= 4
                # e.g.: A=5, B=5, C=4, D=1 → remove D
                #        A=3, B=3, C=3, D=3 → keep all
                if min_vote <= 1 and max_vote >= 4:
                    weak_opts = [opt for opt in voted_answers if option_votes[opt] == min_vote]
                    if len(weak_opts) == 1:
                        voted_answers.discard(weak_opts[0])
                        print(f"[Logic Check] Removed weak option '{weak_opts[0]}' (votes: {min_vote} vs max: {max_vote})")
                # Otherwise keep all 4 options (trust voting result)
        
        vote_summary = ", ".join(f"{opt}:{count}" for opt, count in sorted(option_votes.items()))
        print(f"\n[Vote counts] {vote_summary}")
        #print(f"[Threshold {self.vote_threshold}] Selected: {sorted(voted_answers)}")
        print(f"[Threshold: general={self.vote_threshold}, D={self.d_option_threshold}] Selected: {sorted(voted_answers)}")
        
        # If none pass threshold, select those with highest votes
        if not voted_answers:
            max_votes = max(option_votes.values())
            if max_votes > 0:
                voted_answers = {opt for opt, count in option_votes.items() 
                               if count == max_votes}
        
        # ============ STAGE 2: Targeted verification (optional) ============
        # Find "borderline options" (votes near threshold)
        uncertain_options = {opt for opt, count in option_votes.items() 
                           if 1 < count < self.vote_threshold}
        
        if uncertain_options:
            print(f"\n[Verification] Uncertain options: {sorted(uncertain_options)}")
            # Targeted verification logic can be added here
        
        # ============ Post-processing ============
        final_answers = post_process_answers(voted_answers, item.options)
        
        # Build output
        output = f"""
            ========== SELF-CONSISTENCY (Option-Level Voting) ==========
            Samples: {self.num_samples}, Threshold: {self.vote_threshold}
            Vote counts: {vote_summary}
            Voted answers: {sorted(voted_answers)}
            After post-processing: {sorted(final_answers)}

            ========== BEST RESPONSE ==========
            {all_responses[0] if all_responses else "No response"}

            Final Answer I Reasoned: {",".join(sorted(final_answers)) if final_answers else "A"}
            """
        return output
