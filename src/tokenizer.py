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


def _decode_token(token_str: str, unicode_to_byte: dict[str, int]) -> str:
    """
    Converts one raw BPE token string back into real text.
    """
    byte_list = [unicode_to_byte[c] for c in token_str]
    byte_str = bytes(byte_list)
    return byte_str.decode('utf-8', errors='replace')


def decode(token_ids: list[int],
           id_to_token: list[str | None],
           unicode_to_byte: dict[str, int]) -> str:
    """
    Decodes a token id sequence into text using our own byte-level BPE
    reverse mapping, joining raw tokens before decoding UTF-8.
    """
    raw_string = ""
    for token_id in token_ids:
        token_str = id_to_token[token_id]
        if token_str is not None:
            raw_string += token_str
    return _decode_token(raw_string, unicode_to_byte)
