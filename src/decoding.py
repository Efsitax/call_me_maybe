from llm_sdk import Small_LLM_Model
import numpy as np
from typing import Callable


def _mask_logits(logits: list[float], valid_ids: set[int]) -> np.ndarray:
    """
    Sets every logit outside valid_ids to -inf, leaving valid ones untouched.
    """
    masked_logits: np.ndarray = np.full(len(logits), -np.inf)
    for i in valid_ids:
        masked_logits[i] = logits[i]
    return masked_logits


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


def select_candidates(model: Small_LLM_Model,
                      prompt: str,
                      candidates: list[str],
                      id_to_text: list[str | None],
                      max_steps: int = 30,
                      encode_fn: (
                          Callable[[str, Small_LLM_Model], list[int]] | None
                      ) = None
                      ) -> str:
    """
    Runs constrained decoding step by step until the generated text
    exactly matches one of the given candidates, then returns it.
    """
    if not candidates:
        raise ValueError("Candidates list must not be empty.")
    if encode_fn is not None:
        input_ids: list[int] = encode_fn(prompt, model)
    else:
        input_ids = model.encode(prompt)[0].tolist()
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


def _is_valid_number_prefix(text: str) -> bool:
    """
    Checks whether text could still be extended into a valid JSON number.
    """
    if text == "":
        return True
    allowed_chars = set("0123456789.-")
    for ch in text:
        if ch not in allowed_chars:
            return False
    if text.count(".") > 1:
        return False
    if text.count("-") > 1:
        return False
    if "-" in text and not text.startswith("-"):
        return False
    if "." in text:
        digits_before_dot = text.split(".")[0].lstrip("-")
        if digits_before_dot == "":
            return False
    return True


def _is_complete_number(text: str) -> bool:
    """
    Checks whether text is already a complete, valid JSON number.
    """
    if text == "" or text == "-":
        return False
    if text.endswith("."):
        return False
    return _is_valid_number_prefix(text) and text[-1].isdigit()


def _valid_number_token_ids(current_text: str,
                            id_to_text: list[str | None]) -> set[int]:
    """
    Determines which token ids can be appended to current_text while still
    forming a valid (partial) JSON number.
    """
    valid_ids: set[int] = set()
    for token_id, token_text in enumerate(id_to_text):
        if token_text is None:
            continue
        next_text = current_text + token_text
        if _is_valid_number_prefix(next_text):
            valid_ids.add(token_id)
    return valid_ids


def generate_number(model: Small_LLM_Model,
                    prompt: str,
                    id_to_text: list[str | None],
                    stop_token_id: int,
                    max_steps: int = 12,
                    forbidden_values: set[float] | None = None,
                    max_attempts: int = 2) -> float:
    """
    Runs constrained decoding to produce a JSON number, retrying with
    a different first token if the result collides with
    forbidden_values.
    """
    if forbidden_values is None:
        forbidden_values = set()
    excluded_first_ids: set[int] = set()
    value: float = 0.0

    for _ in range(max_attempts):
        input_ids: list[int] = model.encode(prompt)[0].tolist()
        current_text: str = ""
        last_chosen_id: int | None = None
        repeat_count: int = 0
        first_chosen_id: int | None = None
        try:
            for _ in range(max_steps):
                logits: list[float] = model.get_logits_from_input_ids(
                    input_ids
                )
                valid_ids: set[int] = _valid_number_token_ids(current_text,
                                                              id_to_text)
                if current_text == "":
                    valid_ids -= excluded_first_ids
                if _is_complete_number(current_text):
                    valid_ids.add(stop_token_id)
                masked: np.ndarray = _mask_logits(logits, valid_ids)
                chosen_id: int = int(np.argmax(masked))
                if first_chosen_id is None:
                    first_chosen_id = chosen_id
                if chosen_id == last_chosen_id:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_chosen_id = chosen_id
                if repeat_count >= 2 and _is_complete_number(current_text):
                    break
                if chosen_id == stop_token_id:
                    break
                input_ids.append(chosen_id)
                next_token_text: str | None = id_to_text[chosen_id]
                if next_token_text is None:
                    raise RuntimeError(
                        f"Token id {chosen_id} decoded to None unexpectedly"
                    )
                current_text += next_token_text
            else:
                raise RuntimeError(
                    f"Could not complete a number within {max_steps} steps"
                )
        except IndexError as e:
            raise RuntimeError(
                f"Token id out of range while masking logits: {e}"
            ) from e
        if not _is_complete_number(current_text):
            raise RuntimeError(
                f"Generated text {current_text!r} is not a valid "
                "JSON number."
            )
        assert first_chosen_id is not None
        value = float(current_text)
        if value not in forbidden_values:
            return value
        excluded_first_ids.add(first_chosen_id)
    return value


def generate_string(model: Small_LLM_Model,
                    prompt: str,
                    id_to_text: list[str | None],
                    stop_token_id: int,
                    max_steps: int = 30) -> str:
    """
    Runs constrained decoding to produce an open-ended JSON string
    value for the prompt.
    """
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    current_text: str = ""
    valid_ids: set[int] = {i for i, t in enumerate(id_to_text)
                           if t is not None}
    valid_ids.add(stop_token_id)
    try:
        for _ in range(max_steps):
            logits: list[float] = model.get_logits_from_input_ids(input_ids)
            masked: np.ndarray = _mask_logits(logits, valid_ids)
            chosen_id: int = int(np.argmax(masked))
            if chosen_id == stop_token_id:
                break
            next_token_text: str | None = id_to_text[chosen_id]
            if next_token_text is None:
                raise RuntimeError(
                    f"Token id {chosen_id} decoded to None unexpectedly"
                )
            if '"' in next_token_text:
                break
            input_ids.append(chosen_id)
            current_text += next_token_text
        else:
            raise RuntimeError(
                f"Could not complete a string within {max_steps} steps"
            )
    except IndexError as e:
        raise RuntimeError(
            f"Token id out of range while masking logits: {e}"
        ) from e
    return current_text
