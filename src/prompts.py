"""
Centralized prompt storage for all approaches.
Each prompt has a unique name with 'intro', 'system' and 'user' fields.

IMPORTANT: Optimized for SemEval 2026 Task 12 evaluation metric:
- 1.0 points: Perfect match (P = G)
- 0.5 points: Partial match (P ⊂ G, no wrong selections)
- 0.0 points: Any wrong selection or empty answer

Key insight: Wrong selection is severely penalized (0 points),
while missing some correct answers still gives partial credit (0.5 points).
Therefore, CONSERVATIVE strategy is optimal: "Better to miss one than select wrong one"
"""

PROMPTS = {
    # ============================================================
    # Conservative strategy
    # Only choose options with high confidence
    # ============================================================
    "conservative": {
        "intro": "Conservative strategy optimized for partial matching metric. Prioritizes precision over recall.",
        "system_prompt": """
        You are an expert in causal reasoning and abductive inference.

        CRITICAL EVALUATION RULE:
        - Selecting ANY wrong option = 0 points (complete failure)
        - Missing some correct options = 0.5 points (partial credit)
        - Perfect match = 1.0 points

        STRATEGY: Be CONSERVATIVE. Only select options you are HIGHLY CONFIDENT about.
        It is much better to miss one correct answer than to include one wrong answer.
        """,
        "user_prompt": 
        """
        TARGET EVENT:
        {event}

        EVIDENCE DOCUMENTS:
        {docs_text}

        CANDIDATE CAUSES:
        {options_text}

        === CONSERVATIVE ANALYSIS ===

        STEP 1: EVIDENCE CHECK
        For each option, find DIRECT evidence in documents:
        - Option A: [Quote from Doc if exists] or "NO DIRECT EVIDENCE"
        - Option B: [Quote from Doc if exists] or "NO DIRECT EVIDENCE"
        - Option C: [Quote from Doc if exists] or "NO DIRECT EVIDENCE"
        - Option D: [Quote from Doc if exists] or "NO DIRECT EVIDENCE"

        STEP 2: CONFIDENCE ASSESSMENT
        Rate each option's confidence level:
        - HIGH: Direct documentary evidence + clear temporal precedence + obvious causal mechanism
        - MEDIUM: Some evidence but uncertain causation or timing
        - LOW: Weak/no evidence, speculation, or likely a consequence not cause

        STEP 3: STRATEGIC SELECTION (Remember: wrong = 0 points, partial = 0.5 points)
        - HIGH confidence → SELECT
        - MEDIUM confidence → DO NOT SELECT (risk not worth it)
        - LOW confidence → DO NOT SELECT

        STEP 4: SPECIAL CASES
        1. DUPLICATE OPTIONS: If two options have identical/nearly identical text, select ALL or NONE of them together.
        2. "NONE CORRECT" OPTION: Select ONLY if ALL other options are LOW confidence. Never select alongside other causes.

        STEP 5: FINAL VERIFICATION
        Before answering, ask yourself: "Am I CERTAIN about each selection? Any doubt = don't select."

        === OUTPUT ===
        State your final answer as: "Final Answer I Reasoned: X" or "Final Answer I Reasoned: X,Y,Z"

        CRITICAL:
        1. At least one answer is required (empty answer = 0 points)
        2. Only select HIGH confidence options
        3. "Final Answer I Reasoned: ..." must be the LAST line
        """
    },

    # ============================================================
    # Evidence anchoring strategy
    # Each choice must be supported by clear evidence
    # ============================================================
    "evidence_anchored": {
        "intro": "Evidence-anchored approach requiring explicit document citations for each selection.",
        "system_prompt": """
        You are an expert in evidence-based causal reasoning.

        CORE PRINCIPLE: No evidence, no selection.
        Every selected option MUST have explicit supporting evidence from the documents.

        EVALUATION CONTEXT:
        - Wrong selection = 0 points (catastrophic)
        - Missing correct = 0.5 points (acceptable)
        - Perfect = 1.0 points (ideal)
        """,
        "user_prompt": """
        TARGET EVENT:
        {event}

        EVIDENCE DOCUMENTS:
        {docs_text}

        CANDIDATE CAUSES:
        {options_text}

        === EVIDENCE-ANCHORED ANALYSIS ===

        For EACH option, complete this analysis:

        **Option A**: {first_option_text}
        - Evidence: [QUOTE exact text from document] or "NONE FOUND"
        - Temporal: Does evidence show this happened BEFORE target event? [YES/NO/UNCLEAR]
        - Causal Link: Does this CAUSE or ENABLE the target event? [YES/NO]
        - Verdict: [SELECT - with evidence] or [REJECT - insufficient evidence]

        **Option B**: (same format)
        **Option C**: (same format)  
        **Option D**: (same format)

        === SELECTION RULES ===
        1. ONLY select options with ALL of:
        - Explicit documentary evidence (quoted)
        - Temporal precedence (YES)
        - Causal link (YES)

        2. If options have identical text → select all or none together

        3. "None of the others" option:
        - Select ONLY if all other options lack evidence
        - NEVER select alongside other causes

        === OUTPUT ===
        Final Answer I Reasoned: [Only options meeting ALL criteria above]

        CRITICAL: "Final Answer I Reasoned: ..." must be the absolute LAST line.
        """
    },

    # ============================================================
    # Balanced strategy
    # Make a balance between radical and conservative
    # ============================================================
    "balanced": {
        "intro": "Balanced strategy weighing precision and recall based on evidence strength.",
        "system_prompt": """
        You are an expert in causal reasoning with document evidence.

        EVALUATION METRIC:
        - Perfect match = 1.0 points
        - Correct subset (no errors) = 0.5 points  
        - Any wrong selection = 0.0 points

        ⚠️ CRITICAL RULES:
        1. Over-selection is the #1 cause of failure (53% of errors)
        2. Option D is especially problematic - be EXTRA cautious
        3. When in doubt, DO NOT select
        4. Better to get 0.5 points (partial) than 0 points (wrong)

        ⚠️ CAUSAL CHAIN WARNING:
        - Is option X a DIRECT cause of the target event?
        - Or is option X a CONSEQUENCE of another option?
        - Or is option X just CORRELATED but not causal?

        Example: If the event is "Biden declared disaster for Texas"
        - ✅ "Winter storm hit Texas" = DIRECT CAUSE
        - ❌ "Power plants shut down" = CONSEQUENCE of the storm
        - ❌ "ERCOT acknowledged cold weather" = Just correlated

        ⚠️ "NONE CORRECT" DETECTION:
        - If NO option has strong documentary evidence as a DIRECT cause
        - If all options are consequences, correlations, or background info
        - Then select the "None of the others are correct" option
        - Don't force a selection if nothing truly fits!

        ⚠️ COMMON MISTAKES TO AVOID:
        1. Selecting background/historical events that are NOT direct causes
        2. Selecting multiple options when only one is the TRUE cause
        3. Confusing "happened before" with "caused"

        Only select DIRECT causes with STRONG documentary evidence!
        """,
    "user_prompt": """
        TARGET EVENT:
        {event}

        EVIDENCE DOCUMENTS:
        {docs_text}

        CANDIDATE CAUSES:
        {options_text}

        === STRICT ANALYSIS FRAMEWORK ===

        **STEP 1: Evidence Check**
        For each option, quote DIRECT evidence from documents:
        | Option | Evidence Quote | Found? |
        |--------|---------------|--------|
        | A      | "..." or NONE | Yes/No |
        | B      | "..." or NONE | Yes/No |
        | C      | "..." or NONE | Yes/No |
        | D      | "..." or NONE | Yes/No |

        **STEP 2: Causation Test (Only for options with evidence)**
        For each option with evidence, answer:
        1. Did this happen BEFORE the target event? (Yes/No)
        2. Did this DIRECTLY CAUSE the target event? (Yes/No)
        3. Is this a consequence or just correlated? (Cause/Consequence/Correlated)

        **STEP 3: Final Selection**
        - Select ONLY options where: Evidence=Yes AND Before=Yes AND DirectCause=Yes
        - If unsure about ANY option, DO NOT select it
        - Prefer selecting fewer options over risking wrong selection

        === OUTPUT ===
        Final Answer I Reasoned: [Your selections]

        CRITICAL: This line must be the LAST line of your response.
        """
    },
    "cot": {
        "intro": "The basic zero-shot CoT approach",
        "system_prompt": "You are an expert detective and logic analyst. Your task is Abductive Reasoning: identifying the most plausible cause for an event based on incomplete evidence.",
        "user_prompt": """
        Target Event:
        {event}

        Retrieved Evidence:
        {docs_text}

        Candidate Causes:
        {options_text}

        Instruction:
        1. Analyze the relationship between the event and the documents.
        2. Evaluate each candidate cause.
        3. Select the plausible cause(s).

        Output format:
        First, provide a detailed reasoning chain explaining:
        1. What information is found in the documents related to the event.
        2. How each candidate cause relates to the event.
        3. Why certain causes are more plausible than others.
        4. Which documents support or contradict each option.
        5. Your final conclusion with clear justification.
        Finally, state the answer strictly in this format: "Final Answer I Reasoned: [Option Label]".
        Your output must strictly adhere to the format and order specified above!!!

        Note that there may be one or multiple correct option(s), you have to select ALL options that are directly supported or strongly implied by the documents as plausible causes of the event, for example the final answer you reasoned is A:
        1. if you find B and C have the same content with A, then you have to output A,B,C.
        2. if you find B express the same meaning with A but just with a different way of saying it, then you have to output A,B.
        3. if you find C encompassed by A, then you have to output A,C.
        
        If there is an option states "None of the others are correct causes." and you have clear evidence that NONE of other options are plausible causes according to what you've retrieved, then choose only this one.

        CRITICAL:: 
        1. There is guaranteed to be AT LEAST one correct answer from the given options, so an empty answer is NOT allowed!
        2. The "Final Answer I Reasoned: ..." line MUST be the very last line of your response. Do NOT write anything after it!
        """
    }
}