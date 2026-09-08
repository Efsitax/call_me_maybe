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


def _load_tokenizer_json(model: Small_LLM_Model) -> dict[str, Any]:
    """
    Loads and parses the model's tokenizer.json file.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    tokenizer_json_path = vocab_path.parent / "tokenizer.json"
    with open(tokenizer_json_path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def _load_raw_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """
    Loads the vocabulary from tokenizer.json's model.vocab field,
    which has the same shape regardless of the underlying tokenizer
    algorithm (BPE, Unigram, WordPiece, ...), unlike the native
    vocab file whose format varies by model.
    """
    token_data = _load_tokenizer_json(model)
    vocab = token_data["model"]["vocab"]
    if isinstance(vocab, dict):
        return {token: int(token_id) for token, token_id in vocab.items()}
    return {entry[0]: i for i, entry in enumerate(vocab)}


def _load_added_tokens(model: Small_LLM_Model) -> list[dict[str, Any]]:
    """
    Loads added tokens.
    """
    token_data = _load_tokenizer_json(model)
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
            decoded = model.decode([i])
            id_to_text[i] = decoded if decoded else None
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


def find_special_token_id(model: Small_LLM_Model, content: str) -> int:
    """
    Finds the token id of a special token by its literal content
    (e.g. "<|im_end|>").
    """
    added_token: list[dict[str, Any]] = _load_added_tokens(model)
    for entry in added_token:
        if entry["content"] == content:
            return int(entry["id"])
    else:
        raise ValueError(f"Special token {content!r} not found.")


def find_stop_token_id(model: Small_LLM_Model) -> int:
    """
    Finds the model's end-of-turn token id by trying common special
    token names in order (ChatML, GPT-2 style, then Llama/Mistral
    style), so different models can be supported without hardcoding
    one specific token.
    """
    candidates: list[str] = ["<|im_end|>", "<|endoftext|>", "</s>"]
    stop_token: int = -1
    for candidate in candidates:
        try:
            stop_token = find_special_token_id(model, candidate)
            break
        except ValueError:
            continue
    if stop_token != -1:
        return stop_token
    else:
        raise ValueError("Stop token not found.")
