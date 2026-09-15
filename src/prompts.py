from llm_sdk import Small_LLM_Model
from src import FunctionDefinition
from .vocab import find_special_token_id
import json
from typing import Any

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
                            func_def: FunctionDefinition) -> str:
    """
    Builds a prompt showing the function as a Python signature and
    asking for the call that fulfils the question. A code context makes
    the model write argument *values* (e.g. "*", "[aeiou]") rather than
    the words that describe them. The question comes last so the model
    doesn't echo nearby prompt words instead of answering.
    """
    signature = ", ".join(f"{name}: {param.type}"
                          for name, param in func_def.parameters.items())
    return ("```python\n" +
            f"def {func_def.name}({signature}):\n" +
            f"    \"\"\"{func_def.description}\"\"\"\n" +
            "```\n" +
            f"Write the Python call to {func_def.name} " +
            f"for this request: {question}")


def _python_literal(value: Any) -> str:
    """
    Renders an already-resolved parameter value as Python source.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _parameter_prefill(func_def: FunctionDefinition,
                       param_name: str,
                       resolved: dict[str, Any]) -> str:
    """
    Primes the assistant's reply with the call written up to the
    requested argument, including the values already resolved for the
    preceding parameters, and an opening quote for strings.
    """
    param_type = func_def.parameters[param_name].type
    previous = "".join(f"{name}={_python_literal(value)}, "
                       for name, value in resolved.items())
    opening_quote = "\"" if param_type == "string" else ""
    return ("```python\n" +
            f"{func_def.name}({previous}{param_name}={opening_quote}")


def build_parameter_prompt(model: Small_LLM_Model,
                           question: str,
                           func_def: FunctionDefinition,
                           param_name: str,
                           resolved: dict[str, Any] | None = None) -> str:
    """
    Builds the chat-wrapped, prefilled prompt for a parameter's value.
    resolved holds the values already generated for earlier parameters
    of the same call, in order.
    """
    if resolved is None:
        resolved = {}
    raw_prompt = _build_parameter_prompt(question, func_def)
    prefill = _parameter_prefill(func_def, param_name, resolved)
    return _wrap_chat_prompt(model, raw_prompt, prefill)
