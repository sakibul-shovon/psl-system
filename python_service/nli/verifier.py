"""
Local NLI (Natural Language Inference) verifier.

Model: cross-encoder/nli-deberta-v3-small (~85MB, CPU, ~50ms per pair)

Given a (premise, hypothesis) pair it returns one of:
  ENTAILMENT    — hypothesis is supported by the premise
  NEUTRAL       — premise neither supports nor contradicts
  CONTRADICTION — hypothesis contradicts the premise

CRITICAL: label order is read from model.config.id2label at load time.
Different NLI checkpoints assign different integer indices to each label.
Hardcoding index 0 = ENTAILMENT silently inverts grounding on the wrong model.
"""

import logging
import threading
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_model: Optional[CrossEncoder] = None
_label_map: dict[int, str] = {}   # index → label name, populated at load time
_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model, _label_map
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading NLI model '%s'...", MODEL_NAME)
                _model = CrossEncoder(MODEL_NAME, device="cpu", automodel_args={"low_cpu_mem_usage": False})
                # Read label order from the model config — never hardcode
                id2label = _model.model.config.id2label
                _label_map = {int(k): v.upper() for k, v in id2label.items()}
                logger.info("NLI label map (CRITICAL — verify this): %s", _label_map)
                for idx, label in _label_map.items():
                    if "ENTAIL" in label:
                        logger.info("ENTAILMENT is at index %d", idx)
    return _model


def predict(premise: str, hypothesis: str) -> str:
    """
    Run NLI on one (premise, hypothesis) pair.

    Args:
        premise:    the evidence chunk content.
        hypothesis: one sentence from the generated draft.

    Returns:
        "ENTAILMENT", "NEUTRAL", or "CONTRADICTION"
    """
    model = _get_model()
    scores = model.predict([(premise, hypothesis)])  # shape: (1, num_labels)
    predicted_idx = int(scores[0].argmax())
    label = _label_map.get(predicted_idx, "NEUTRAL")
    return label


def predict_batch(pairs: list[tuple[str, str]]) -> list[str]:
    """
    Run NLI on multiple (premise, hypothesis) pairs in one batch.
    More efficient than calling predict() in a loop.
    """
    if not pairs:
        return []
    model = _get_model()
    scores = model.predict(pairs)
    return [_label_map.get(int(row.argmax()), "NEUTRAL") for row in scores]
