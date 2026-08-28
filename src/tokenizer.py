from typing import Any
from llm_sdk import Small_LLM_Model
from pathlib import Path
import json
import re


def _byte_to_unicode() -> dict[int, str]:
    """
    Maps each byte to a printable unicode character (GPT-2 BPE style).
    """
    bs = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('¡'), ord('¬') + 1))
        + list(range(ord('®'), ord('ÿ') + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for c in cs]
    return dict(zip(bs, chars))


def _unicode_to_byte() -> dict[str, int]:
    """
    Builds the reverse mapping: character back to its byte value.
    """
    byte_to_unicode = _byte_to_unicode()
    return {char: byte for byte, char in byte_to_unicode.items()}


def _load_tokenizer_json(model: Small_LLM_Model) -> dict[str, Any]:
    """
    Loads and parses the model's tokenizer.json file.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    tokenizer_json_path = vocab_path.parent / "tokenizer.json"
    with open(tokenizer_json_path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def _load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """
    Loads the token string to id mapping from vocab.json.
    """
    with open(model.get_path_to_vocab_file()) as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def _build_id_to_token(model: Small_LLM_Model,
                       vocab: dict[str, int]) -> list[str | None]:
    """
    Builds an id-indexed lookup covering both vocab.json and any added
    special tokens (e.g. <|im_end|>).
    """
    added_tokens: list[dict[str, Any]] = _load_tokenizer_json(model).get(
        "added_tokens", []
    )
    vocab_size = max(
        [*vocab.values()] + [entry["id"] for entry in added_tokens]
    ) + 1
    id_to_token: list[str | None] = [None] * vocab_size
    for token_str, token_id in vocab.items():
        id_to_token[token_id] = token_str
    for entry in added_tokens:
        id_to_token[entry["id"]] = entry["content"]
    return id_to_token


def _decode_token(token_str: str, unicode_to_byte: dict[str, int]) -> str:
    """
    Converts one raw BPE token string back into real text.
    """
    byte_list = [unicode_to_byte[c] for c in token_str]
    byte_str = bytes(byte_list)
    return byte_str.decode('utf-8', errors='replace')


def _decode_ids(token_ids: list[int],
                id_to_token: list[str | None],
                unicode_to_byte: dict[str, int]) -> str:
    """
    Decodes a token id sequence into text, joining raw tokens first so
    multi-byte characters spanning token boundaries decode correctly.
    """
    raw_string = ""
    for token_id in token_ids:
        token_str = id_to_token[token_id]
        if token_str is not None:
            raw_string += token_str
    return _decode_token(raw_string, unicode_to_byte)


def decode(token_ids: list[int], model: Small_LLM_Model) -> str:
    """
    Decodes token ids into text, building the required lookup tables
    from the model itself.
    """
    vocab = _load_vocab(model)
    id_to_token = _build_id_to_token(model, vocab)
    unicode_to_byte = _unicode_to_byte()
    return _decode_ids(token_ids, id_to_token, unicode_to_byte)


def _load_merge_ranks(model: Small_LLM_Model) -> dict[tuple[str, str], int]:
    """
    Loads BPE merge rules from tokenizer.json as a {pair: priority} map,
    where a lower number means the merge should happen earlier.
    """
    token_data = _load_tokenizer_json(model)
    merges: list[list[str]] = token_data["model"]["merges"]
    merge_ranks: dict[tuple[str, str], int] = {}
    for order, pair in enumerate(merges):
        merge_ranks[(pair[0], pair[1])] = order
    return merge_ranks


def _load_split_pattern(model: Small_LLM_Model) -> re.Pattern[str]:
    """
    Reads the model's own pre-tokenization regex from tokenizer.json
    and adapts \\p{L}/\\p{N} (unsupported by Python's re) to
    equivalent built-in patterns.
    """
    token_data = _load_tokenizer_json(model)
    pre_tokenizers = token_data["pre_tokenizer"]["pretokenizers"]
    regex = None
    for pre_tokenizer in pre_tokenizers:
        if pre_tokenizer["type"] == "Split":
            regex = pre_tokenizer["pattern"]["Regex"]
            break
    if regex is None:
        raise ValueError("Could not find a Split pre-tokenizer pattern.")
    regex = regex.replace(r"[^\r\n\p{L}\p{N}]", r"[^\r\n\w]")
    regex = regex.replace(r"[^\s\p{L}\p{N}]", r"[^\s\w]")
    regex = regex.replace(r"\p{L}", r"[^\W\d_]")
    regex = regex.replace(r"\p{N}", r"\d")
    return re.compile(regex)


def _merge_word(chars: list[str],
                merge_ranks: dict[tuple[str, str], int]) -> list[str]:
    """
    Repeatedly merges the highest-priority adjacent pair in chars
    until no known merge applies, returning the final token pieces.
    """
    while len(chars) > 1:
        pairs: list[tuple[str, str]] = []
        for i in range(len(chars) - 1):
            pairs.append((chars[i], chars[i + 1]))
        found_ranks: dict[tuple[str, str], int] = {
            pair: merge_ranks[pair] for pair in pairs if pair in merge_ranks
        }
        if not found_ranks:
            break
        best_pair = min(found_ranks, key=lambda p: found_ranks[p])
        new_chars: list[str] = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                new_chars.append(chars[i] + chars[i + 1])
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars

    return chars


def _encode_piece(piece: str, vocab: dict[str, int]) -> list[int]:
    """
    Looks up piece's token id, falling back to encoding it one base
    character at a time if the merged piece itself isn't in vocab.
    """
    if piece in vocab:
        return [vocab[piece]]
    ids: list[int] = []
    for ch in piece:
        if ch not in vocab:
            raise ValueError(f"Character {ch!r} not found in vocab.")
        ids.append(vocab[ch])
    return ids


def encode(text: str, model: Small_LLM_Model) -> list[int]:
    """
    Encodes text into token ids using our own byte-level BPE
    implementation (pre-tokenization + iterative merging + vocab lookup).
    """
    vocab = _load_vocab(model)
    split_pattern = _load_split_pattern(model)
    merge_ranks = _load_merge_ranks(model)
    byte_to_unicode = _byte_to_unicode()
    token_ids: list[int] = []
    for chunk in split_pattern.findall(text):
        chars = [byte_to_unicode[b] for b in chunk.encode('utf-8')]
        pieces = _merge_word(chars, merge_ranks)
        for piece in pieces:
            token_ids.extend(_encode_piece(piece, vocab))
    return token_ids
