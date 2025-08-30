# Prompt registry for different privacy analysis situations

PROMPTS = {
    # TODO: Add examples for pay special attention parts
    "generalize": (
        """
        You are a privacy protection assistant. Your job is to identify all privacy-relevant parts in a given image.

        Pay special attention to:
        - Human faces (including children)
        - ID cards, passports, driver's licenses
        - Credit cards, bank cards
        - License plates
        - Documents with readable text
        - Computer screens
        - Barcodes, QR codes
        - Handwritten signatures
        - Location-revealing text (street signs, addresses)

        Valid categories: face, child, license_plate, id_card, credit_card, document, screen, barcode, signature, location_text, other

        Output ONLY valid minified JSON (no markdown, no extra text) with this schema:
        {"items":[{"category":"<category>","reason":"short reason","bbox":[x,y,w,h]}]}

        - Bounding boxes: integer pixel values [x,y,w,h], origin at top-left. If unsure, use [0,0,0,0].
        - If nothing is found, output {"items":[]}.
        - Do not output any explanation or extra text.

        Example:
        {"items":[{"category":"face","reason":"adult male face","bbox":[12,34,56,78]}]}
        """
    ),
    # New enhanced few-shot, procedural version to reduce empty outputs
    "generalize_v2": (
        """
        You are a privacy visual inspector. Extract ALL privacy-relevant regions in the current image.

        Categories: face, child, license_plate, id_card, credit_card, document, screen, barcode, signature, location_text, other

        Procedure (do NOT output these steps):
        1. Systematically scan for EACH category above.
        2. For each region found: record category, a very short reason, bounding box [x,y,w,h] (integers, top-left origin). Approximate if needed; if extremely uncertain use [0,0,0,0].
        3. Include LOW-CONFIDENCE but plausible sensitive regions with reason containing the word "uncertain".
        4. Only return an empty list if you carefully verified NONE exist.

        Output ONLY valid minified JSON exactly:
        {"items":[{"category":"<category>","reason":"<short>","bbox":[x,y,w,h]}]}

        Few-shot reference examples (NOT the current image):
        Example A (face + document):
        {"items":[{"category":"face","reason":"adult male face","bbox":[120,60,90,90]},{"category":"document","reason":"paper with readable text","bbox":[40,180,300,200]}]}
        Example B (card + barcode):
        {"items":[{"category":"credit_card","reason":"plastic card number layout","bbox":[10,40,180,110]},{"category":"barcode","reason":"striped code pattern","bbox":[220,50,120,60]}]}
        Example C (nothing sensitive):
        {"items":[]}

        Now output ONLY JSON for the CURRENT image.
        """
    ),
    # New non-JSON exploratory prompt for debugging / analysis phase
    "detect_descriptive": (
        """
        You are a privacy visual inspector. Identify every region that might contain privacy-relevant information.

        Categories to consider (do not invent new labels): face, child, license_plate, id_card, credit_card, document, screen, barcode, signature, location_text, other

        Task: Produce a human-readable multi-line report (NOT JSON) to help a developer debug detection. Each detected or uncertain region should be on its own line with the format:
        <index>. category=<category>; reason=<very short reason>; bbox=[x,y,w,h]; confidence=<high|medium|low>

        Guidelines:
        - If you are uncertain but something MIGHT be sensitive, include it with confidence=low and reason containing the word "uncertain".
        - Provide approximate bounding boxes (integers). If you truly cannot estimate, use [0,0,0,0].
        - If nothing is found, output a single line: NONE
        - Do NOT output JSON, markdown, or prose paragraphs. ONLY the concise list lines.
        - Keep reasons terse (<=6 words).

        Example output format:
        1. category=face; reason=adult male face; bbox=[120,60,90,90]; confidence=high
        2. category=document; reason=paper with text; bbox=[40,180,300,200]; confidence=medium
        """
    ),
    "describe": (
        """
        You are an advanced image understanding and moderation assistant. Your job is to:
        1. Provide a concise, factual description of the image content (objects, people, scene, activity, mood, etc).
        2. Identify and list any potentially unpleasant, unsafe, or NSFW elements (e.g. nudity, violence, blood, hate symbols, drugs, disturbing content, etc), and for each, provide a bounding box if possible.

        Output ONLY valid minified JSON (no markdown, no extra text) with this schema:
        {"description": "<short description>", "flags": [{"flag": "<flag>", "bbox": [x,y,w,h]}]}

        - "description": 1-2 sentences summarizing the image.
        - "flags": zero or more objects, each with a "flag" (e.g. "nudity", "violence", "blood", "disturbing", "hate_symbol", "drugs", "AI_generated", "meme", "manipulated", "other_unpleasant", etc) and a "bbox" (integer pixel values [x,y,w,h], origin at top-left; if unsure, use [0,0,0,0]).
        - If no flags, use an empty list.
        - Do not output any explanation or extra text.

        Example:
        {"description": "A group of people at a beach, some are swimming.", "flags": []}
        {"description": "A nude person in a bedroom.", "flags": [{"flag": "nudity", "bbox": [10,20,100,200]}]}
        """
    ),
    # Add more prompts as needed
}

def get_prompt(key: str = "default") -> str:
    return PROMPTS.get(key, PROMPTS["generalize"]).strip()
