from llm_sdk import Small_LLM_Model
import json
from pathlib import Path
from typing import Any
import sys


def _get_vocab_size(model: Small_LLM_Model) -> int:
    """
    Determines the model's vocabulary size.
    """
    ids = model.encode("42")
    logits = model.get_logits_from_input_ids(ids[0].tolist())
    return len(logits)


def _load_raw_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """
    Loads the raw vocabulary in JSON.
    """
    vocab: dict[str, int]
    with open(model.get_path_to_vocab_file()) as f:
        vocab = json.load(f)
    return vocab


def _load_added_tokens(model: Small_LLM_Model) -> list[dict[str, Any]]:
    """
    Loads added tokens.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    tokenizer_json_path = vocab_path.parent / "tokenizer.json"
    with open(tokenizer_json_path) as f:
        token_data = json.load(f)
    added: list[dict[str, Any]] = token_data.get("added_tokens", [])
    return added


def _build_id_to_token(vocab: dict[str, int],
                       added_tokens: list[dict[str, Any]],
                       vocab_size: int) -> list[str | None]:
    """
    Combines the raw vocabulary and added tokens into a single id-indexed list.
    """
    id_to_token: list[str | None] = [None] * vocab_size
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str
    for entry in added_tokens:
        id_to_token[entry["id"]] = entry["content"]
    return id_to_token


def _build_id_to_text(id_to_token: list[str | None],
                      vocab_size: int,
                      model: Small_LLM_Model) -> list[str | None]:
    """
    Decodes each valid token id into its real text representation.
    """
    id_to_text: list[str | None] = [None] * vocab_size
    for i in range(vocab_size):
        if id_to_token[i] is None:
            continue
        try:
            id_to_text[i] = model.decode([i])
        except Exception:
            id_to_text[i] = None
    return id_to_text


def build_vocabulary(model: Small_LLM_Model) -> list[str | None]:
    """
    Builds the full id-to-text vocabulary lookup for the given model.
    """
    try:
        vocab_size: int = _get_vocab_size(model)
        raw_vocab: dict[str, int] = _load_raw_vocab(model)
        added_tokens: list[dict[str, Any]] = _load_added_tokens(model)
        id_to_token: list[str | None] = _build_id_to_token(raw_vocab,
                                                           added_tokens,
                                                           vocab_size)
        id_to_text: list[str | None] = _build_id_to_text(id_to_token,
                                                         vocab_size,
                                                         model)
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Failed at parsing on JSON.")
        sys.exit(1)
    return id_to_text
