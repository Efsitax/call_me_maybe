from llm_sdk import Small_LLM_Model
import numpy as np


def _surviving_candidates(current_text: str,
                          candidates: list[str]) -> list[str]:
    """
    Filters candidates down to those that still start with the given text.
    """
    survived: list[str] = []
    for candidate in candidates:
        if candidate.startswith(current_text):
            survived.append(candidate)
    return survived


def _valid_next_token_ids(current_text: str,
                          candidates: list[str],
                          id_to_text: list[str | None]) -> set[int]:
    """
    Determines which token ids can be appended to current_text while still
    matching at least one candidate.
    """
    valid_ids: set[int] = set()
    for token_id, token_text in enumerate(id_to_text):
        if token_text is None:
            continue
        next_text = current_text + token_text
        survivals = _surviving_candidates(next_text, candidates)
        if survivals:
            valid_ids.add(token_id)
    return valid_ids


def _mask_logits(logits: list[float], valid_ids: set[int]) -> np.ndarray:
    """
    Sets every logit outside valid_ids to -inf, leaving valid ones untouched.
    """
    masked_logits: np.ndarray = np.full(len(logits), -np.inf)
    for i in valid_ids:
        masked_logits[i] = logits[i]
    return masked_logits


def select_candidates(model: Small_LLM_Model,
                      prompt: str,
                      candidates: list[str],
                      id_to_text: list[str | None],
                      max_steps: int = 30) -> str:
    """
    Runs constrained decoding step by step until the generated text
    exactly matches one of the given candidates, then returns it.
    """
    if not candidates:
        raise ValueError("Candidates list must not be empty.")
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    current_text: str = ""
    try:
        for _ in range(max_steps):
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            valid_ids: set[int] = _valid_next_token_ids(current_text,
                                                        candidates,
                                                        id_to_text)
            masked: np.ndarray = _mask_logits(logits, valid_ids)
            chosen_id: int = int(np.argmax(masked))
            input_ids.append(chosen_id)
            next_token_text: str | None = id_to_text[chosen_id]
            if next_token_text is None:
                raise RuntimeError(
                    f"Token id {chosen_id} decoded to None unexpectedly"
                )
            current_text += next_token_text
            if current_text in candidates:
                break
        else:
            raise RuntimeError(
                f"Could not select a candidate within {max_steps} steps"
            )
    except IndexError as e:
        raise RuntimeError(
            f"Token id out of range while masking logits: {e}"
        ) from e
    return current_text
