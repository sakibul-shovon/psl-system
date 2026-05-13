"""
Prompt assembler — builds the full Gemini generation prompt.

The prompt has four sections:
  1. SYSTEM  — role + strict grounding rules
  2. EVIDENCE — [E1]..[E5] blocks with breadcrumb + OCR confidence
  3. PATTERNS — learned style patterns injected here in Phase 6 (empty for now)
  4. TASK     — what type of draft to generate and for which document

Keeping prompt construction in one place means every module that needs
to change the prompt (e.g. pattern injection in Phase 6) edits one file.
"""

from python_service.retrieval.evidence import EvidenceItem

_SYSTEM = """\
You are a legal document analyst for Pearson Specter Litt.
Generate a {draft_type} based EXCLUSIVELY on the evidence provided below.

RULES — follow these without exception:
1. Every factual claim MUST end with an inline citation [E1], [E2], etc.
2. Only cite evidence that directly supports the claim.
3. If evidence is insufficient for a claim, write [INSUFFICIENT EVIDENCE: reason].
4. Do NOT infer, assume, or extrapolate beyond what evidence explicitly states.
5. If two evidence items conflict, report both versions and flag the conflict.
6. Omission is correct. Fabrication is a critical failure.
7. Dates, dollar amounts, party names, and section references must be quoted exactly as they appear in the evidence.

OUTPUT FORMAT — return valid JSON only, no markdown, matching this schema exactly:
{{
  "draftType": "{draft_type}",
  "title": "string",
  "sections": [
    {{
      "sectionId": "sec_1",
      "title": "string",
      "content": "string with inline [E1] citations",
      "citedEvidence": ["E1", "E2"],
      "confidence": "HIGH | MEDIUM | LOW"
    }}
  ],
  "overallConfidence": "HIGH | MEDIUM | LOW",
  "warnings": ["list of any issues, conflicts, or low-confidence notes"]
}}
"""

_EVIDENCE_HEADER = "\n\nEVIDENCE (use ONLY this to make claims):\n"

_PATTERNS_HEADER = "\n\nLEARNED STYLE PATTERNS (apply these to this draft):\n"

_TASK = "\n\nTASK: Generate a {draft_type} for document: {document_title}\nReturn only the JSON object described above."


def build_prompt(
    evidence_items: list[EvidenceItem],
    draft_type: str,
    document_title: str,
    patterns: list[dict] | None = None,
) -> str:
    """
    Build the full generation prompt.

    Args:
        evidence_items: labeled [E1]-[E5] evidence from Phase 3.
        draft_type:     e.g. "case_fact_summary".
        document_title: display name of the source document.
        patterns:       learned style patterns from Phase 6 (None = not yet built).

    Returns:
        Complete prompt string ready to send to Gemini.
    """
    parts: list[str] = []

    # Section 1: system instructions
    parts.append(_SYSTEM.format(draft_type=draft_type))

    # Section 2: evidence blocks
    parts.append(_EVIDENCE_HEADER)
    for item in evidence_items:
        parts.append(item.to_prompt_block())
        parts.append("")   # blank line between evidence items

    # Section 3: learned style patterns (Phase 6 populates this)
    if patterns:
        parts.append(_PATTERNS_HEADER)
        for i, pat in enumerate(patterns, start=1):
            parts.append(
                f"{i}. [{pat.get('ruleType', 'style').upper()}] "
                f"confidence:{pat.get('confidence', 0):.2f}\n"
                f"   Rule: {pat.get('description', '')}\n"
                f"   Before: \"{pat.get('fewShotBefore', '')}\"\n"
                f"   After:  \"{pat.get('fewShotAfter', '')}\""
            )

    # Section 4: task
    parts.append(_TASK.format(draft_type=draft_type, document_title=document_title))

    return "\n".join(parts)
