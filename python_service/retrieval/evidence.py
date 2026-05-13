"""
Evidence packager — formats reranked chunks as [E1]..[E5] items.

The evidence package is what gets passed to Gemini for generation.
Each item gets a label ([E1], [E2], ...) and OCR confidence tier
(HIGH / MEDIUM / LOW) so the LLM can signal uncertainty in its draft.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# OCR confidence tiers — shown in the evidence block so Gemini can signal
# uncertainty in citations from degraded scans.
HIGH_CONF  = 0.85
MEDIUM_CONF = 0.70   # below this = LOW


@dataclass
class EvidenceItem:
    evidence_id: str          # "E1", "E2", ...
    chunk_id: str
    content: str
    breadcrumb: str
    document_title: str
    relevance_score: float    # reranker score
    retrieval_method: str
    ocr_confidence_avg: float
    has_low_conf_regions: bool

    @property
    def confidence_tier(self) -> str:
        if self.ocr_confidence_avg >= HIGH_CONF:
            return "HIGH"
        elif self.ocr_confidence_avg >= MEDIUM_CONF:
            return "MEDIUM"
        else:
            return "LOW"

    def to_prompt_block(self) -> str:
        """
        Format this evidence item as it appears in the Gemini prompt.

        Example output:
          [E2] ARTICLE IV > Section 4.2 Limitation of Liability
               Source: Hardman Group Lease 2024.pdf | OCR: MEDIUM
          Pursuant to Article IV, the Tenant shall not be liable...
        """
        conf_note = ""
        if self.confidence_tier != "HIGH":
            conf_note = f" ⚠ OCR confidence: {self.confidence_tier}"

        header = (
            f"[{self.evidence_id}] {self.breadcrumb}\n"
            f"     Source: {self.document_title}{conf_note}"
        )
        return f"{header}\n{self.content}"

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "breadcrumb": self.breadcrumb,
            "document_title": self.document_title,
            "relevance_score": round(self.relevance_score, 4),
            "retrieval_method": self.retrieval_method,
            "ocr_confidence_avg": self.ocr_confidence_avg,
            "has_low_conf_regions": self.has_low_conf_regions,
            "confidence_tier": self.confidence_tier,
        }


def package_evidence(
    reranked: list[dict],
    document_title: str = "Unknown Document",
) -> list[EvidenceItem]:
    """
    Convert reranked retrieval results into labeled EvidenceItems.

    Args:
        reranked:       top-k results from reranker (chunk_id, rerank_score, payload).
        document_title: display name for the source document.

    Returns:
        List of EvidenceItem, labeled E1, E2, ... in relevance order.
    """
    items = []
    for i, result in enumerate(reranked, start=1):
        payload = result.get("payload", {})
        items.append(EvidenceItem(
            evidence_id=f"E{i}",
            chunk_id=result["chunk_id"],
            content=payload.get("content", ""),
            breadcrumb=payload.get("breadcrumb", result["chunk_id"]),
            document_title=document_title,
            relevance_score=result["rerank_score"],
            retrieval_method="dense+bm25+rerank",
            ocr_confidence_avg=payload.get("ocr_confidence_avg", 1.0),
            has_low_conf_regions=payload.get("has_low_conf_regions", False),
        ))

    logger.debug("Packaged %d evidence items", len(items))
    return items
