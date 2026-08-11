from llm_sdk import Small_LLM_Model
from src import FunctionDefinition
from .vocab import find_special_token_id

_chat_template_cache: dict[int, tuple[bool, str]] = {}


def _get_chat_template_info(model: Small_LLM_Model) -> tuple[bool, str]:
    """
    Determines, once per model, whether it supports chat/think special
    tokens, caching the result to avoid re-reading the vocab files.
    """
    key = id(model)
    if key in _chat_template_cache:
        return _chat_template_cache[key]

    supports_chat = True
    try:
        find_special_token_id(model, "<|im_start|>")
        find_special_token_id(model, "<|im_end|>")
    except ValueError:
        supports_chat = False

    think_prefill = ""
    if supports_chat:
        try:
            find_special_token_id(model, "<think>")
            find_special_token_id(model, "</think>")
            think_prefill = "<think>\n\n</think>\n\n"
        except ValueError:
            pass

    _chat_template_cache[key] = (supports_chat, think_prefill)
    return _chat_template_cache[key]


def _wrap_chat_prompt(model: Small_LLM_Model, content: str,
                      assistant_prefill: str = "") -> str:
    """
    Wraps content in chat format if the model supports it, priming
    the assistant's reply with assistant_prefill.
    """
    supports_chat, think_prefill = _get_chat_template_info(model)
    if not supports_chat:
        return content + assistant_prefill

    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n{think_prefill}{assistant_prefill}")


def _ordinal(position: int) -> str:
    """
    Converts a 1-based position to its English ordinal (1st, 2nd, ...).
    """
    if 10 <= position % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    return f"{position}{suffix}"


def _build_selection_prompt(question: str,
                            fn_defs: list[FunctionDefinition]) -> str:
    """
    Builds a prompt asking the model to pick a function for the question.
    """
    if not fn_defs:
        raise ValueError("fn_defs list must not be empty.")
    fn_str: str = "\n- ".join(f"{fn.name}: {fn.description}" for fn in fn_defs)
    return (f"Question: {question}\n" +
            f"Available functions:\n- {fn_str}\n" +
            "Which function should be called? " +
            "Answer with only the function name:")


def build_function_prompt(model: Small_LLM_Model,
                          question: str,
                          fn_defs: list[FunctionDefinition]) -> str:
    """
    Builds the chat-wrapped prompt for selecting a function.
    """
    selection_prompt = _build_selection_prompt(question, fn_defs)
    return _wrap_chat_prompt(model, selection_prompt)


def _build_parameter_prompt(question: str,
                            func_def: FunctionDefinition,
                            param_name: str) -> str:
    """
    Builds a prompt asking for one parameter's value. The question
    comes last so the model doesn't echo nearby prompt words instead
    of answering.
    """
    param_type = func_def.parameters[param_name].type
    all_params = ", ".join(func_def.parameters.keys())
    return (f"The function \"{func_def.name}\" will be called: " +
            f"{func_def.description}\n" +
            "Its parameters, in order, are: " +
            f"{all_params}\n" +
            f"Extract the value for parameter '{param_name}' " +
            f"(a {param_type}) from the question below.\n" +
            f"Question: {question}")


def _parameter_prefill(func_def: FunctionDefinition, param_name: str) -> str:
    """
    Primes the assistant's reply: an opening quote for strings (so it
    writes the value instead of a sentence), a neutral marker otherwise.
    """
    param_type = func_def.parameters[param_name].type
    if param_type != "string":
        return ">>>"
    param_names = list(func_def.parameters.keys())
    position = param_names.index(param_name) + 1
    ordinal = _ordinal(position)
    return f"The {ordinal} parameter's value is \""


def build_parameter_prompt(model: Small_LLM_Model,
                           question: str,
                           func_def: FunctionDefinition,
                           param_name: str) -> str:
    """
    Builds the chat-wrapped, prefilled prompt for a parameter's value.
    """
    raw_prompt = _build_parameter_prompt(question, func_def, param_name)
    prefill = _parameter_prefill(func_def, param_name)
    return _wrap_chat_prompt(model, raw_prompt, prefill)
