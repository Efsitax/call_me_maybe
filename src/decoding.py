from llm_sdk import Small_LLM_Model
import numpy as np
import re
from typing import Callable


def _mask_logits(logits: list[float], valid_ids: set[int]) -> np.ndarray:
    """
    Sets every logit outside valid_ids to -inf.
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
    Finds token ids that keep current_text a prefix of some candidate.
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
    Runs constrained decoding until the output exactly matches one of
    candidates.
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
    Finds token ids that keep current_text a valid partial JSON number.
    """
    valid_ids: set[int] = set()
    for token_id, token_text in enumerate(id_to_text):
        if token_text is None:
            continue
        next_text = current_text + token_text
        if _is_valid_number_prefix(next_text):
            valid_ids.add(token_id)
    return valid_ids


def _numbers_in_text(text: str) -> set[str]:
    """
    Extracts every number-like substring (e.g. "12", "-3", "0.375")
    literally appearing in text.
    """
    return set(re.findall(r"-?\d+(?:\.\d+)?", text))


def _number_grounded(number_text: str, grounding_pool: set[str]) -> bool:
    """
    True if question has no literal numbers, or number_text matches
    one exactly.
    """
    return not grounding_pool or number_text in grounding_pool


def generate_number(model: Small_LLM_Model,
                    prompt: str,
                    req_type: str,
                    id_to_text: list[str | None],
                    stop_token_id: int,
                    question: str = "",
                    max_steps: int = 12,
                    forbidden_values: set[float] | None = None,
                    max_attempts: int = 10) -> float:
    """
    Runs constrained decoding to produce a JSON number, retrying on a
    new first token when the result collides with forbidden_values or
    isn't grounded in question.
    """
    if forbidden_values is None:
        forbidden_values = set()
    grounding_pool: set[str] = _numbers_in_text(question)
    excluded_first_ids: set[int] = set()
    value: float | int = 0.0

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
                grounded = _number_grounded(current_text, grounding_pool)
                if _is_complete_number(current_text) and grounded:
                    break
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
        if req_type == "number":
            value = float(current_text)
        elif req_type == "integer":
            value = int(current_text)
        grounded = _number_grounded(current_text, grounding_pool)
        if value not in forbidden_values and grounded:
            return value
        excluded_first_ids.add(first_chosen_id)
    return value


_CONTINUATION_CHAR = re.compile(r"[\w/.-]")


def _is_grounded_in_text(value: str, text: str) -> bool:
    """
    True if value appears in text as a whole match, not a truncated
    fragment (e.g. "home/data" inside "/home/data" doesn't count).
    """
    if not value:
        return True
    idx = text.find(value)
    if idx == -1:
        return False
    if idx > 0 and _CONTINUATION_CHAR.match(text[idx - 1]):
        return False
    end = idx + len(value)
    if end < len(text) and _CONTINUATION_CHAR.match(text[end]):
        return False
    return True


def _valid_string_token_ids(id_to_text: list[str | None],
                            stop_token_id: int) -> set[int]:
    """
    Defines the grammar of a string value: any decodable token except
    one containing an escaped backslash ("\\\\"). The prefill opens a
    Python string literal, so without this rule the model writes
    escape sequences (C:\\\\Users) instead of the raw value (C:\\Users).
    """
    valid_ids: set[int] = set()
    for token_id, token_text in enumerate(id_to_text):
        if token_text is None or "\\\\" in token_text:
            continue
        valid_ids.add(token_id)
    valid_ids.add(stop_token_id)
    return valid_ids


def _string_needs_retry(result: str, question: str) -> bool:
    """
    True only when result is a truncated fragment of a match in
    question; a clean match or an absent (derived) value need no
    retry.
    """
    if not question or _is_grounded_in_text(result, question):
        return False
    return result in question


def generate_string(model: Small_LLM_Model,
                    prompt: str,
                    id_to_text: list[str | None],
                    stop_token_id: int,
                    question: str = "",
                    max_steps: int = 30,
                    max_attempts: int = 5) -> str:
    """
    Runs constrained decoding to produce an open-ended JSON string,
    retrying on a new first token when the result is a truncated
    fragment of question.
    """
    excluded_first_ids: set[int] = set()
    result: str = ""
    first_result: str | None = None

    for _ in range(max_attempts):
        input_ids: list[int] = model.encode(prompt)[0].tolist()
        current_text: str = ""
        first_chosen_id: int | None = None
        valid_ids: set[int] = _valid_string_token_ids(id_to_text,
                                                      stop_token_id)
        try:
            for _ in range(max_steps):
                logits: list[float] = model.get_logits_from_input_ids(
                    input_ids
                )
                ids = (valid_ids - excluded_first_ids if current_text == ""
                       else valid_ids)
                masked: np.ndarray = _mask_logits(logits, ids)
                chosen_id: int = int(np.argmax(masked))
                if first_chosen_id is None:
                    first_chosen_id = chosen_id
                if chosen_id == stop_token_id:
                    break
                next_token_text: str | None = id_to_text[chosen_id]
                if next_token_text is None:
                    raise RuntimeError(
                        f"Token id {chosen_id} decoded to None unexpectedly"
                    )
                if '"' in next_token_text:
                    current_text += next_token_text.split('"')[0]
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
        result = current_text.strip()
        if not _string_needs_retry(result, question):
            return result
        if first_result is None:
            first_result = result
        assert first_chosen_id is not None
        excluded_first_ids.add(first_chosen_id)
    return first_result if first_result is not None else result
