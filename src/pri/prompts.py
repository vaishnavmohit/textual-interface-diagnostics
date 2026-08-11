"""Prompts for the natural-image benchmarks, ported verbatim from
code/prompts/natural/single.py (frozen COLM provenance) so `pri` reproduces the
inherited results.

Prompt *text is provenance*: the describe/reason stages record the SHA-256 of the
exact string used (see docs/REPRODUCIBILITY_DESIGN.md). Do not edit these strings
without bumping the corresponding schema_version — a changed prompt is a
different experiment.

Perception prompts (Stage 1, describe):
    describe_structured   task-blind hierarchical scene/object/relation schema (CA, C2)
    describe_activity     relation-focused schema for Winoground

Reasoner prompts (Stage 2, reason):
    system_eval           reasoner system prompt (Bongard rule induction)
    user_eval(specs,m,n)  reasoner user prompt embedding the descriptions

New E2 condition prompts (C1 flat, C3 task-aware, C4 fixed reinspection) are
added in build increment 11.
"""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Stable provenance hash of a prompt string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1 — perception (description) prompts
# ---------------------------------------------------------------------------

def describe_structured() -> str:
    """Task-blind hierarchical description schema (CA / C2)."""
    return """
            Carefully examine the provided image and identify all possible visual elements, organizing them into a detailed hierarchical structure. Start with broad categories and progress to more specific subcategories. This should cover everything visible in the image, ensuring no detail is overlooked. Structure your findings in a JSON format to enable easy comparison and synthesis of data from other images. This will help discern patterns, contexts, and rules valuable for identifying or understanding query images.

            Your hierarchy might encompass the following elements:

            1. **Scene/Environment**: Description of the overall setting depicted, such as urban, natural, indoor, or outdoor scenes.
            2. **Objects**: Define distinct items or entities present in the scene.
            - **Living Beings**: Animals, humans, or other biological entities.
                - Species or classification (e.g., dog, bird, human).
                - Characteristics (e.g., color, posture, movement).
            - **Inanimate Objects**: Both synthetic and natural elements.
                - Categories (e.g., vehicle, building, trees).
                - Properties (e.g., color, size, material, shape).
            3. **Activities**: Observable actions or interactions involving any objects or beings.
            - Specific descriptions of actions (e.g., walking, flying).
            - Participants involved in these actions.
            4. **Contextual Elements**: Environmental conditions and time markers, such as time of day or weather.
            - Detailed characteristics (e.g., cloudy, night, winter).
            5. **Visual Patterns**: Prominent colors, textures, and patterns that are visually significant.
            6. **Emotional Undertones**: Any emotional presence or expressions evident in the image.
            7. **Textual Information**: Any visible text within the image, including what it says and its visual style.
            8. **Summary**: A concise narrative summarizing the overall content and context of the image.

            Ensure that every aspect from the image is represented under these categories. The information should be presented in the following JSON format:

            {
            "Scene": {
                "Description": "..."
            },
            "Objects": {
                "Living Beings": [...],
                "Inanimate Objects": [...]
            },
            "Activities": [...],
            "Contextual Elements": {
                "Time of Day": "...",
                "Weather": "..."
            },
            "Visual Patterns": {
                "Dominant Colors": [...],
                "Textures": [...]
            },
            "Emotional Undertones": "..."
            "Textual Information": "..."
            "Summary": "..."
            }
            Ensure that the JSON output is clear, well-formatted, and free of unnecessary explanations. Omit the ```json tags at the beginning and end of the page.
            """


def describe_activity() -> str:
    """Relation-focused description schema for Winoground (entities get IDs so
    the reasoner can bind who-does-what-to-whom)."""
    return """
            Carefully examine the provided image, focusing intently on the actions, interactions, and relationships between entities. Your goal is to produce a detailed, structured description in JSON format that captures the nuances often present in comparative image pairs like those in Winoground.

            Identify all visual elements and organize them hierarchically. Pay special attention to the 'Activities' and 'Spatial Relationships' sections as they define how entities relate to each other.

            Your hierarchy must encompass the following elements:

            1.  **Scene/Environment**: Describe the overall setting (e.g., park, kitchen, street).
            2.  **Objects**: Define distinct items or entities. Assign a unique simple ID (e.g., H1 for human 1, D1 for dog 1, T1 for table 1) to each distinct entity for cross-referencing.
                *   **Living Beings**: Humans, animals. Include species/type, key characteristics (color, clothing, posture, apparent age/gender for humans).
                *   **Inanimate Objects**: Furniture, vehicles, tools, natural elements. Include category, key properties (color, material, state).
            3.  **Activities/Interactions**: Detail observable actions and interactions, as these define the dynamic **relationships** between entities. Be specific about **who (actor ID) is doing what (action) to whom (target ID)**, plus manner/context and any alternative interpretations.
            4.  **Spatial Relationships**: Describe key static positional **relationships** between entities using their IDs (e.g., 'on', 'under', 'next to', 'facing', 'left of', 'behind').
            5.  **Contextual Elements**: Time of day, weather, lighting conditions.
            6.  **Visual Patterns**: Dominant colors, textures.
            7.  **Emotional Undertones/Expressions**: Apparent emotions or facial expressions of living beings (use IDs).
            8.  **Textual Information**: Any visible text.
            9.  **Overall Summary**: A brief narrative summarizing the main event or state depicted in the image, referencing key entities by ID if helpful.

            Structure your findings strictly in the following JSON format, using the assigned IDs consistently:

            {
                "Scene": { "Description": "..." },
                "Objects": {
                    "Living Beings": [ ],
                    "Inanimate Objects": [ ]
                },
                "Activities": [ ],
                "Spatial Relationships": [ ],
                "Contextual Elements": { "Time of Day": "...", "Weather": "...", "Lighting": "..." },
                "Visual Patterns": { "Dominant Colors": [...], "Textures": [...] },
                "Emotional Undertones": "...",
                "Textual Information": "...",
                "Summary": "..."
            }

            Ensure the JSON is valid, clear, and contains no explanatory text outside the JSON structure itself. Omit the ```json markdown tags. Focus on objective visual details and the precise nature of interactions.
            """


# ---------------------------------------------------------------------------
# Stage 2 — reasoner prompts (Bongard rule induction)
# ---------------------------------------------------------------------------

def system_eval() -> str:
    return """
        You are an expert in visual reasoning and analysis, working with complex datasets such as the Bongard dataset. Your task is to analyze descriptions of images from two categories (cat_2 and cat_1) along with a test image, provided in JSON format.

        The goal is to:
        1. Identify the **common characteristics or patterns** in the positive samples (cat_2) that distinctly separate them from the negative samples (cat_1).
        2. Derive a **clear and concise rule** that defines the positive samples.
        3. Apply this rule to the test image and categorize it as belonging to either cat_1 or cat_2.

        Ensure that the JSON output is clear, well-formatted, and free of unnecessary explanations.
        Omit the ```json tags at the beginning and end of the page. The format of your output should be as follows:
        - Analysis: Provide a brief analysis of the distinguishing characteristics or patterns.
        - Rule: State the rule that separates cat_2 from cat_1.
        - Test Image: Analyze the test image based on the derived rule.
        - Conclusion: Clearly state whether the test image belongs to cat_1 or cat_2.

        Respond in the format provided, and avoid unnecessary explanations or text.
        """


def user_eval(all_image_specs, m: int, n: int) -> str:
    return f"""
        We are working with the Bongard dataset, which contains {m} images in cat_2 (positive samples) and {n} images in cat_1 (negative samples). These categories are defined as follows:
        - Cat_2: Positive samples that follow some common rule.
        - Cat_1: Negative samples that may not follow any specific rule.

        The image descriptions for the positive samples, negative samples, and the test image are provided in JSON format. Analyze the common patterns or characteristics in the cat_2 samples that distinguish them from cat_1 samples.

        Your task is to:
        1. Derive the rule that defines the cat_2 samples.
        2. Apply this rule to categorize the test image.

        Here are the image descriptions:

        ### Positive Samples (cat_2):
        {all_image_specs[:m]}

        ### Negative Samples (cat_1):
        {all_image_specs[m:m+n]}

        ### Test Image:
        {all_image_specs[-1]}

        Ensure that the JSON output is clear, well-formatted, and free of unnecessary explanations.
        Omit the ```json tags at the beginning and end of the page. The format of your output should be as follows:

        {{
        "Analysis": (Your analysis here)
        "Rule": (The distinguishing rule here)
        "Test Image": (Test image details)
        "Conclusion": (cat_1 or cat_2)
        }}
        """


# ---------------------------------------------------------------------------
# Registry: which perception prompt each context uses. schema_version ties the
# frozen description artifact to the exact prompt (bump on any text change).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# E2 perception conditions. Each is a DISTINCT schema_version so its frozen
# artifact can never be confused with the task-blind CA descriptions.
#   C1 ca_flat      : free-form description (no structured schema), task-blind
#   C3 ca_taskaware : SAME structured schema as C2 + a task block that reveals
#                     the image's role (positive/negative/query). This
#                     deliberately breaks task-blindness — it is the quantity the
#                     C3-C2 contrast measures. The schema body is byte-identical
#                     to C2 so the contrast isolates task-conditioning, not
#                     structure.
# ---------------------------------------------------------------------------

def describe_flat() -> str:
    """C1: task-blind free-form (unstructured) description."""
    return """
            Carefully examine the provided image and describe everything you can see in a few clear sentences of plain prose. Cover the overall scene, all salient objects and their attributes (colour, size, material, shape), any people or animals and what they are doing, spatial relationships between things, notable colours and textures, and any visible text. Do not use JSON or any structured format — write natural prose. Aim to be thorough enough that someone who cannot see the image could reconstruct its contents.
            """


def describe_taskaware(role: str) -> str:
    """C3: task-aware structured description. `role` in {positive, negative, query}.

    Prepends a task block (which reveals that this is a Bongard concept-learning
    problem and this image's role) to the *identical* C2 structured schema.
    """
    task_block = f"""
            You are describing one image from a Bongard visual concept-learning problem. In such a problem, the POSITIVE examples all share a single hidden rule (a common concept), and the NEGATIVE examples do not satisfy that rule; a query image must then be judged against the rule. This image's role in the problem is: **{role}**.

            With that task in mind, produce a description that pays particular attention to features that could plausibly define or violate such a hidden rule. Use exactly the structured schema specified below.
            """
    return task_block + describe_structured()


def describe_relational() -> str:
    """C2r: the C2 schema plus an explicit relational field. Still task-blind.

    Motivated by a measured leak rather than a guess. On Bongard-HOI the action
    that defines a concept is named in only 61-82% of positive images'
    descriptions under the C2 schema, and spuriously in 20-55% of negatives, so
    the interface transmits the discriminating evidence noisily even though the
    schema nominally has an Activities field. Its JSON template is a bare list,
    which invites summary actions rather than per-pair relations.

    This variant adds one field asking for each agent-object pair explicitly:
    what the agent does to the object, whether they are in contact, and the
    role each plays. It remains task-blind in the sense the ladder requires --
    no mention of the task, the rule, the categories, or the image's role -- so
    C2r vs C2 isolates SCHEMA ELICITATION, where C3 vs C2 isolated task
    knowledge. Everything else in the schema is carried over unchanged.
    """
    extra = """
            In addition to the categories above, include one further top-level field:

            9. **Interactions**: for EVERY pair consisting of a living being and an object
            (or two living beings) that are related in the image, one entry giving:
            - "Agent": which being acts;
            - "Target": which object or being is acted upon;
            - "Action": the specific action the agent performs on the target, as a verb
              (for example: riding, holding, carrying, repairing, pushing, feeding,
              standing beside, looking at). Name the action even when it is ordinary.
            - "Contact": whether the agent physically touches the target ("contact",
              "no contact", or "unclear");
            - "Role": what part the target plays in the action (for example: being
              ridden, being held, merely nearby, being looked at).

            Include an entry for every such pair you can see, including pairs where the
            relation is incidental or the agent is merely near the target. If no living
            being is present, use an empty list.

            Add this field to the JSON as:
            "Interactions": [
                {"Agent": "...", "Target": "...", "Action": "...", "Contact": "...", "Role": "..."}
            ]
            """
    return describe_structured() + extra


PERCEPTION_PROMPTS = {
    "ca":          {"fn": describe_structured, "schema_version": "structured_v1"},
    "winoground":  {"fn": describe_activity,   "schema_version": "activity_v1"},
    # json_output=False: this condition deliberately asks for prose. Constraining
    # the API to emit a JSON object would contradict the prompt and yield empty
    # responses -- the whole point of C1 is unstructured text.
    "ca_flat":     {"fn": describe_flat,       "schema_version": "flat_v1",
                    "json_output": False},
    "ca_relational": {"fn": describe_relational, "schema_version": "relational_v1"},
    "ca_taskaware":{"fn": describe_taskaware,  "schema_version": "taskaware_v1",
                    "task_aware": True},
    # ca_summary is DERIVED, never generated: scripts/derive_summary_descriptions.py
    # extracts the Summary field from the structured artifact. Registered here only
    # so the reason stage can resolve its schema_version; the prompt entry records
    # that the text originated from describe_structured. json_output is False
    # because the stored text is prose.
    "ca_summary":  {"fn": describe_structured, "schema_version": "summary_v1",
                    "json_output": False},
}


def fixed_reinspect_question() -> str:
    """C4: the single preregistered, benchmark-generic re-inspection question.
    Fixed (not reasoner-generated) — this is what makes C4 the budget-matched
    control for ICA's adaptive query."""
    return ("List in detail every object, attribute, action, and spatial "
            "relationship visible in this image, including small, background, or "
            "easily-missed details.")


def ca_reinspect_final_prompt(initial_decision, reinspect_answer: str) -> str:
    """C4/reinspection final: re-decide given the initial CA analysis plus a fresh
    detailed re-inspection of the query image."""
    import json as _json
    return f"""You previously analysed a Bongard visual concept-learning problem from image descriptions and reached a tentative decision. A fresh, detailed re-inspection of the query image is now provided. Reconsider all the evidence and give your final decision.

    **Your previous analysis (JSON):**
    {_json.dumps(initial_decision, indent=2)}

    **Fresh re-inspection of the query image:**
    {reinspect_answer}

    Return your response strictly in the following JSON format. Omit the ```json tags.
    {{
    "Analysis": (updated analysis incorporating the re-inspection),
    "Rule": (the distinguishing rule),
    "Test Image": (query image details),
    "Conclusion": (cat_1 or cat_2)
    }}
    """


def wino_reinspect_final_prompt(initial_decision, reinspect_answer) -> str:
    """C4 (Winoground) final: re-decide given the initial CA analysis plus the
    answer to the FIXED re-inspection question.

    The Winoground counterpart of :func:`ca_reinspect_final_prompt`. It must NOT
    reuse ica_prompts.text_score_final_ica_prompt: that prompt tells the model to
    consider "the reasoning behind any question you asked", which is false here —
    the question is preregistered, not reasoner-chosen. Saying so would leak
    adaptivity into the control and contaminate the C5-C4 contrast.
    """
    import json as _json
    return f"""You previously analysed a caption-to-image matching problem from image descriptions and reached a tentative decision. A fresh, standard re-inspection of the image is now provided. This re-inspection answers a fixed, preregistered question — it was not chosen by you and is not tailored to your analysis. Reconsider all the evidence and give your final decision.

    **Your previous analysis (JSON):**
    {_json.dumps(initial_decision, indent=2)}

    **Fresh re-inspection of the image:**
    {_json.dumps(reinspect_answer, indent=2) if isinstance(reinspect_answer, (dict, list)) else reinspect_answer}

    Return your response strictly in the following JSON format. Do not include any text outside the JSON.
    {{
        "analysis": (updated analysis incorporating the re-inspection),
        "category": ('cat_0' or 'cat_1')
    }}
    """


def role_for_index(idx: int, m: int, n: int) -> str:
    """Map an image's position to its Bongard role (positives, then negatives,
    then the query as the last image)."""
    if idx < m:
        return "positive"
    if idx < m + n:
        return "negative"
    return "query"


# ---------------------------------------------------------------------------
# End-to-end multi-image prompts (DVRL / DRL) — ported from
# code/prompts/natural/multi_test.py (DVRL) and multi.py (DRL). No perception
# stage: the VLM sees the images directly. Same BongardDecision JSON output.
# ---------------------------------------------------------------------------

def dvrl_prompt(m: int, n: int) -> str:
    """DVRL: single pass over all m+n support images plus the query."""
    return f"""
    You are provided with {m + n + 1} images: the first {m} samples are `cat_2`, the next {n} samples are `cat_1`, and the last image is the `test image`.
    Analyze the common characteristics or patterns found in the `cat_2` samples (positive samples: following some common rule) that distinctly separate them from the `cat_1` samples (negative samples: it might not follow any possible rule).
    Your task is to:

    1. Determine the rule or criterion that distinguishes the `cat_2` samples from the `cat_1` ones.
    2. Analyse the `test image` (last image).
    3. Provide your conclusion for the `test image` if it can be categorized as either `cat_1` or `cat_2` based on the analysis and the rule.

    Ensure that the JSON output is clear, well-formatted, and free of unnecessary explanations.
    Omit the ```json tags at the beginning and end of the page. The format of your output should be as follows:

    {{
    "Analysis": (Your analysis here)
    "Rule": (The distinguishing rule here)
    "Test Image": (Test image details)
    "Conclusion": (cat_1 or cat_2)
    }}
    """


def drl_rule_prompt(m: int, n: int) -> str:
    """DRL stage 1: derive the rule from the support set only (no query image)."""
    return f"""
    You are provided with {m + n} images: the first {m} samples are `cat_2` (positive samples following some common rule), the next {n} samples are `cat_1` (negative samples).
    Analyze the common characteristics or patterns found in the `cat_2` samples that distinctly separate them from the `cat_1` samples.
    Provide the rule that defines the `cat_2` samples, and at the end write a "summary" of the rule in fewer than 20 words.
    Ensure the output is clear and free of unnecessary explanations. Omit the ```json tags.
    """


def drl_text_rule_prompt(all_image_specs, m: int, n: int) -> str:
    """Textual-DRL stage 1: derive the rule from the support DESCRIPTIONS only.

    The matched control for CA - DRL. `drl_rule_prompt` shows the model the
    support IMAGES; this shows it the frozen support DESCRIPTIONS and nothing
    else. Everything the two share is held byte-identical -- the task framing,
    the cat_2/cat_1 vocabulary, and above all the "fewer than 20 words" rule
    budget, which is what makes the compression comparable. The query is
    withheld at this stage exactly as it is in DRL.
    """
    return f"""
    You are provided with descriptions of {m + n} images in JSON format: the first {m} samples are `cat_2` (positive samples following some common rule), the next {n} samples are `cat_1` (negative samples).
    Analyze the common characteristics or patterns found in the `cat_2` samples that distinctly separate them from the `cat_1` samples.
    Provide the rule that defines the `cat_2` samples, and at the end write a "summary" of the rule in fewer than 20 words.
    Ensure the output is clear and free of unnecessary explanations. Omit the ```json tags.

    ### Positive Samples (cat_2):
    {all_image_specs[:m]}

    ### Negative Samples (cat_1):
    {all_image_specs[m:m+n]}
    """


def drl_text_apply_prompt(query_spec, m: int, n: int, summary: str) -> str:
    """Textual-DRL stage 2: apply the derived rule to the query DESCRIPTION.

    Mirrors `drl_apply_prompt` with the query image replaced by its frozen
    description. The support descriptions are NOT re-supplied: the rule summary
    is the only channel from stage 1 to stage 2, which is precisely the
    compression that distinguishes this condition from CA.
    """
    return f"""
    We are working with the Bongard dataset ({m} images in cat_2, {n} in cat_1). The rule distinguishing cat_2 from cat_1 is summarized as:
    {summary}

    Consider this rule and classify the `test image`, whose description is given below, as either cat_1 or cat_2.

    ### Test Image:
    {query_spec}

    Ensure the JSON output is clear and free of unnecessary explanations. Omit the ```json tags. Use the format:

    {{
    "Analysis": (Your analysis here)
    "Rule": (The distinguishing rule here)
    "Test Image": (Test image details)
    "Conclusion": (cat_1 or cat_2)
    }}
    """


def text_score_prompt(image_description: str, caption_0: str, caption_1: str) -> str:
    """Winoground text score: given one image description, choose the caption it
    matches better (cat_0 / cat_1). Ported verbatim from
    code/prompts/natural/winoground.py."""
    return f"""You are provided with a detailed JSON description of a single image and two different captions (Caption 0 and Caption 1). Your task is to evaluate how well the image description matches *each* caption individually, determine which caption provides a stronger match (higher possibility), and explain why. Apply commonsense reasoning where needed.

    **Image Description (JSON):**
    ```json
    {image_description}```

    **Caption 0:** "{caption_0}"
    **Caption 1:** "{caption_1}"

    **Instructions:**
    1.  **Deconstruct Image Description:** Identify the main entities (using `id`s), actions (`Activities`), attributes (`characteristics`, `properties`), and relationships (`Spatial Relationships`) detailed in the JSON description. Use commonsense to understand the full context implied by the description.
    2.  **Evaluate Match with Caption 0:** Systematically check how well the key elements identified in Caption 0 (entities, actions, attributes, relationships) are supported by the details in the `Image Description` JSON. Assess the overall strength of the match and note discrepancies.
    3.  **Evaluate Match with Caption 1:** Perform the same systematic check and assessment against Caption 1.
    4.  **Compare Matches and Conclude:** Compare the strength of the match for Caption 0 versus Caption 1 and explain which the description matches with higher possibility.
    5.  **Categorize:** Assign 'cat_0' if the image description matches Caption 0 better, or 'cat_1' if it matches Caption 1 better.

    Return your response strictly in the following JSON format:
    {{
        "analysis": (Your detailed analysis comparing the match strength for each caption),
        "category": ('cat_0' or 'cat_1')
    }}

    Do not include any text outside of the JSON structure.
    """


def image_score_prompt(caption: str, image_0_description: str, image_1_description: str) -> str:
    """Winoground image score: given one caption, choose the image description it
    matches better (cat_0 / cat_1). Ported verbatim."""
    return f"""You are provided with a single caption and detailed JSON descriptions of two different images (Image 0 and Image 1). Your task is to evaluate how well the caption matches *each* image description individually, determine which description provides a stronger match (higher possibility), and explain why. Apply commonsense reasoning where needed.

    **Caption**: "{caption}"

    **Image 0 Description (JSON):**
    ```json
    {image_0_description}```

    **Image 1 Description (JSON):**
    ```json
    {image_1_description}```

    **Instructions:**
    1.  **Deconstruct Caption:** Identify the main entities, actions, attributes, and relationships mentioned in the caption. Use commonsense to understand the full context implied by the caption.
    2.  **Evaluate Match with Image 0:** Systematically check how well the caption's key elements are supported by the details in `Image 0 Description`. Assess the overall strength and note discrepancies.
    3.  **Evaluate Match with Image 1:** Perform the same systematic check against `Image 1 Description`.
    4.  **Compare Matches and Conclude:** Compare Image 0 versus Image 1 and explain which description matches the caption with higher possibility.
    5.  **Categorize:** Assign 'cat_0' if the caption matches Image 0 Description better, or 'cat_1' if it matches Image 1 Description better.

    Return your response strictly in the following JSON format:
    {{
        "analysis": (Your detailed analysis comparing the match strength for each description),
        "category": ('cat_0' or 'cat_1')
    }}

    Do not include any text outside of the JSON structure.
    """


def mm_text_score_prompt(caption_0: str, caption_1: str) -> str:
    """Winoground DVRL text score: the image is sent via the multimodal API
    channel (not embedded in text — the legacy code embedded a PIL repr, which
    was noise); pick the caption the image matches better. Ported/cleaned."""
    return f"""You are provided with a single image and two different captions (Caption 0 and Caption 1). Evaluate how well the image matches *each* caption, decide which is the stronger match, and explain why. Apply commonsense reasoning.

    **Caption 0:** "{caption_0}"
    **Caption 1:** "{caption_1}"

    Steps: deconstruct the image; evaluate the match with Caption 0; evaluate the match with Caption 1; compare and conclude; then categorize — 'cat_0' if the image matches Caption 0 better, else 'cat_1'.

    Return strictly this JSON:
    {{
        "analysis": (your comparison),
        "category": ('cat_0' or 'cat_1')
    }}
    Do not include any text outside the JSON.
    """


def mm_image_score_prompt(caption: str) -> str:
    """Winoground DVRL image score: two images sent via the multimodal API
    channel; pick the image the caption matches better."""
    return f"""You are provided with a single caption and two images (Image 0 and Image 1). Evaluate how well the caption matches *each* image, decide which is the stronger match, and explain why. Apply commonsense reasoning.

    **Caption**: "{caption}"

    Steps: deconstruct the caption; evaluate the match with Image 0; evaluate the match with Image 1; compare and conclude; then categorize — 'cat_0' if the caption matches Image 0 better, else 'cat_1'.

    Return strictly this JSON:
    {{
        "analysis": (your comparison),
        "category": ('cat_0' or 'cat_1')
    }}
    Do not include any text outside the JSON.
    """


def drl_apply_prompt(m: int, n: int, summary: str) -> str:
    """DRL stage 2: apply the derived rule to the query image."""
    return f"""
    We are working with the Bongard dataset ({m} images in cat_2, {n} in cat_1). The rule distinguishing cat_2 from cat_1 is summarized as:
    {summary}

    Consider this rule and classify the `test image` as either cat_1 or cat_2.
    Ensure the JSON output is clear and free of unnecessary explanations. Omit the ```json tags. Use the format:

    {{
    "Analysis": (Your analysis here)
    "Rule": (The distinguishing rule here)
    "Test Image": (Test image details)
    "Conclusion": (cat_1 or cat_2)
    }}
    """
