"""ICA (interactive) Winoground prompts, vendored verbatim from
code/prompts/natural/winoground.py so pri reproduces the interactive results.
Do not edit without bumping an ICA schema_version.

Flow per direction: *_ica_prompt asks the reasoner for {analysis,
image_question, category}; if image_question is set, get_visual_answer_ica_prompt
is answered by RE-INSPECTING the real image; *_final_ica_prompt makes the final
call using that answer.
"""

import json

def text_score_ica_prompt(image_description, caption_0, caption_1):
    '''
    Generates a prompt for an LLM to determine which caption better matches 
    an image description by evaluating each match individually and comparing them.

    Args:
        image_description (str): The JSON string description of the image.
        caption_0 (str): The first caption to evaluate.
        caption_1 (str): The second caption to evaluate.

    Returns:
        str: The formatted prompt string.
    '''
    prompt = f"""You are provided with an image description and two captions. Your task is to evaluate how well *each* caption matches the image description individually, determine which caption provides a stronger match, and explain why. Apply commonsense reasoning where needed.

    **Image Description:**
    ```
    {image_description}```

    **Caption 0**: "{caption_0}"
    **Caption 1**: "{caption_1}"

    **Instructions:**
    1.  **Deconstruct Captions:** Identify the main entities, actions, attributes, and relationships mentioned in each caption. Use commonsense to understand the full context implied by each caption.
    2.  **Evaluate Match with Caption 0:** Systematically check how well the key elements identified in Caption 0 are supported by the details in the image description.
    3.  **Evaluate Match with Caption 1:** Perform the same systematic check and assessment against Caption 1.
    4.  **Compare Matches and Conclude:** Compare the strength of the match assessed for Caption 0 versus Caption 1. Explain *why* one caption represents the image with a higher possibility or accuracy than the other. Highlight the specific details (or lack thereof) from the image description that lead to this conclusion. Explicitly mention where commonsense was applied during the evaluation or comparison. 
    5.  **Generate Critical Question (if needed):** If the image description is insufficient to make a definitive determination, formulate one specific, targeted question about the image that, if answered, would provide the exact visual detail needed to resolve the ambiguity. The question should be direct and avoid leaking the answer.
    6.  **Explain Question Impact:** When generating a question, explicitly explain how different possible answers to this question would affect your final categorization decision. Describe what answer would support Caption 0 vs Caption 1, and how this would change your analysis.
    7.  **Categorize:** Assign 'cat_0' if Caption 0 has a higher possibility of matching the image, or 'cat_1' if Caption 1 has a higher possibility of matching the image.

    **IMPORTANT:** If you generate a question, your subsequent analysis will be updated based on the visual answer to that question. The answer will either:
    - Confirm your current categorization and strengthen the evidence
    - Challenge your current categorization and potentially flip the result
    - Provide additional context that refines your confidence level

    Return your response strictly in the following JSON format:
    {{
        "analysis": (Your detailed analysis comparing the match strength for each caption against the image description, explaining why one is a better fit, and noting the use of commonsense),
        "image_question": (Specific targeted question about the image, or none if description is sufficient),
        "question_impact": (Explanation of how different answers to the question would affect the final categorization - include this field only if image_question is not null),
        "category": ('cat_0' or 'cat_1')
    }}

    Do not include any text outside of the JSON structure. Your decision must be based on evaluating the match between each caption and the image description, then comparing those evaluations.
    """
    return prompt


def text_score_final_ica_prompt(previous_response, qa_response):
    '''
    Generates a prompt for final text score analysis after receiving answers to targeted questions.

    Args:
        previous_response (dict): The parsed previous analysis JSON containing analysis, question, and category.
        qa_response (dict): The parsed Q&A response.

    Returns:
        str: The formatted prompt string.
    '''
    
    prompt = f"""You are provided with your previous caption-to-image matching analysis and additional information obtained through a targeted question. Your task is to make a final determination based on all available information. Apply commonsense reasoning where needed.

    **Your Previous Response:**
    ```json
    {json.dumps(previous_response, indent=2)}
    ```
    Additional Information Gathered:
    ```json
    {json.dumps(qa_response, indent=2)}
    ```

    Instructions:
    1. Review Previous Analysis: Consider your initial assessment and the reasoning behind any question you asked.
    2. Incorporate New Information: Evaluate how the answer helps resolve any ambiguity or confirms your initial assessment.
    3. Update Analysis: Provide an updated analysis that incorporates both the original image description and the new answer.
    4. Make Final Decision: Determine your final category based on the complete information.
    
    Return your response strictly in the following JSON format: 
    {{ 
        "analysis": (Your updated detailed analysis incorporating all available information explaining why one is a better fit, and noting the use of commonsense), 
        "category": ('cat_0' or 'cat_1') 
    }}

    Do not include any text outside of the JSON structure. 
    """ 
    return prompt


def image_score_ica_prompt(caption, image_0_description, image_1_description):
    '''
    Generates a prompt for an LLM to determine if a caption has a higher possibility
    of matching image_0_description or image_1_description by evaluating each match
    individually and comparing them, using detailed JSON descriptions and commonsense reasoning.

    Args:
        caption (str): The caption to evaluate.
        image_0_description (str): The JSON string description of the first image.
        image_1_description (str): The JSON string description of the second image.

    Returns:
        str: The formatted prompt string.
    '''
    # Ensure descriptions are treated as strings.
    # The calling code should json.dumps() the dicts before passing.
    prompt = f"""You are provided with a single caption and detailed JSON descriptions of two different images (Image 0 and Image 1). Your task is to evaluate how well the caption matches *each* image description individually, determine which description provides a stronger match (higher possibility), and explain why. Apply commonsense reasoning where needed.

    **Caption**: "{caption}"

    **Image 0 Description (JSON):**
    ```json
    {image_0_description}```

    **Image 1 Description (JSON):**
    ```json
    {image_1_description}```

    **Instructions:**
    1.  **Deconstruct Caption:** Identify the main entities, actions, attributes, and relationships mentioned in the caption (e.g., "old person", "kisses", "young person"). Use commonsense to understand the full context implied by the caption.
    2.  **Evaluate Match with Image 0:** Systematically check how well the key elements identified in the caption are supported by the details in `Image 0 Description`.
        *   Look for specific `id`s, `characteristics`, `actor_ids`, `target_ids`, `action` descriptions, `relationship` types, etc., in the JSON that align with the caption's elements.
        *   Use commonsense reasoning to map caption terms to JSON details (e.g., "old person" might correspond to `characteristics` like "elderly").
        *   Assess the overall strength of the match (e.g., "strong support", "partial support", "weak support", "contradiction"). Note any discrepancies.
    3.  **Evaluate Match with Image 1:** Perform the same systematic check and assessment against `Image 1 Description`.
        *   Look for specific JSON details supporting or contradicting the caption's elements.
        *   Use commonsense reasoning.
        *   Assess the overall strength of the match for Image 1. Note any discrepancies.
    4.  **Compare Matches and Conclude:** Compare the strength of the match assessed for Image 0 versus Image 1. Explain *why* one description represents the caption with a higher possibility or accuracy than the other. Highlight the specific JSON details (or lack thereof) from *both* descriptions that lead to this conclusion. Explicitly mention where commonsense was applied during the evaluation or comparison.
    5.  **Generate Critical Questions (if needed):** If the descriptions are insufficient to make a definitive determination, formulate one specific, targeted question for each image that (ask question without referring or specifying image 0 or image 1), if answered, would provide the exact visual detail needed to resolve the ambiguity. The questions should be direct and avoid leaking the answer.
    6.  **Explain Question Impact:** When generating questions, explicitly explain how different possible answers to these questions would affect your final categorization decision. Describe what answers would support each image choice and how this would change your analysis.
    7.  **Categorize:** Assign 'cat_0' if the caption has a higher possibility of matching Image 0 Description, or 'cat_1' if it has a higher possibility of matching Image 1 Description.

    **IMPORTANT:** If you generate questions, your subsequent analysis will be updated based on the visual answers to those questions. The answers will either:
    - Confirm your current categorization and strengthen the evidence
    - Challenge your current categorization and potentially flip the result
    - Provide additional context that refines your confidence level

    Return your response strictly in the following JSON format:
    {{
        "analysis": (Your detailed analysis comparing the match strength for each description against the caption, explaining why one is a better fit, and noting the use of commonsense),
        "image_0_question": (Specific targeted question for Image 0, or null if description is sufficient),
        "image_1_question": (Specific targeted question for Image 1, or null if description is sufficient),
        "question_impact": (Explanation of how different answers to the questions would affect the final categorization - include this field only if either question is not null),
        "category": ('cat_0' or 'cat_1')
    }}

    Do not include any text outside of the JSON structure. Your decision must be based on evaluating the match between the caption and each description, then comparing those evaluations.
    """
    return prompt


def image_score_final_ica_prompt(previous_response, qa_responses):
    '''
    Generates a prompt for final analysis after receiving answers to targeted questions.

    Args:
        previous_response (dict): The parsed previous analysis JSON containing analysis, questions, and category.
        qa_responses (dict): The parsed Q&A responses.

    Returns:
        str: The formatted prompt string.
    '''
    
    prompt = f"""You are provided with your previous analysis and additional information obtained through targeted questions. Your task is to make a final determination based on all available information. Apply commonsense reasoning where needed.

    **Your Previous Response:**
    ```json
    {json.dumps(previous_response, indent=2)}
    ```
    Additional Information Gathered:
    ```json
    {json.dumps(qa_responses, indent=2)}
    ```
    Instructions:

    Review Previous Analysis: Consider your initial assessment and the reasoning behind any questions you asked.
    Incorporate New Information: Evaluate how the answers help resolve any ambiguity or confirm your initial assessment.
    Update Analysis: Provide an updated analysis that incorporates both the original information and the new answers.
    Make Final Decision: Determine your final category based on the complete information.

    Return your response strictly in the following JSON format: 
    {{ 
        "analysis": (Your updated detailed analysis incorporating all available information), 
        "category": ('cat_0' or 'cat_1') 
    }}

    Do not include any text outside of the JSON structure. 
    """ 
    return prompt


def get_visual_answer_ica_prompt(questions, image_description):
    '''
    Generates a prompt for an LLM to answer specific questions about an image
    by examining the visual content directly, complementing the textual description.

    Args:
        questions (str or list): The specific question(s) to answer about the image.
        image_description (str): The textual description of the image for context.

    Returns:
        str: The formatted prompt string.
    '''
    # Handle both single question (string) and multiple questions (list)
    if isinstance(questions, list):
        questions_text = "\n".join([f"- {q}" for q in questions])
    else:
        questions_text = f"- {questions}"
    
    prompt = f"""You are analyzing an image to answer specific questions that could not be definitively resolved from the textual description alone. Your task is to carefully examine the visual content and provide precise, factual answers. Apply commonsense reasoning where needed.

    **Context - Image Description:**
    ```
    {image_description}```

    **Questions to Answer:**
    ```
    {questions_text}```

    **Instructions:**
    1. **Visual Examination:** Carefully observe the image and identify all relevant visual elements that relate to the questions asked.
    
    2. **Answer Each Question:** For each question, provide a direct, specific answer based on what you can visually observe. Focus on:
       - Concrete visual details (colors, shapes, positions, expressions, actions)
       - Spatial relationships between objects/people
       - Specific attributes or characteristics visible in the image
       - Any actions or interactions taking place
    
    3. **Evidence-Based Responses:** Support your answers with specific visual evidence from the image. Describe exactly what you see that leads to your conclusion.

    Provide direct, concise answers based on what you observe in the image. Use the previous description as context but focus on answering the specific questions.

    **Output JSON Format:**
    {{
        "answer": "Your answer here"
    }}
    """
    
    return prompt

