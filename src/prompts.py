from llm_sdk import Small_LLM_Model
from src import FunctionDefinition
from .vocab import find_special_token_id


def _wrap_chat_prompt(model: Small_LLM_Model, content: str) -> str:
    """
    Wraps raw prompt content in a chat-style template if the model
    supports the expected special tokens, otherwise returns the raw
    content unchanged.
    """
    try:
        find_special_token_id(model, "<|im_start|>")
        find_special_token_id(model, "<|im_end|>")
    except ValueError:
        return content

    think_prefill = ""
    try:
        find_special_token_id(model, "<think>")
        find_special_token_id(model, "</think>")
        think_prefill = "<think>\n\n</think>\n\n"
    except ValueError:
        pass

    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n{think_prefill}")


def _ordinal(position: int) -> str:
    """
    Converts a 1-based position into its English ordinal form
    (1 -> "1st", 2 -> "2nd", 11 -> "11th", 21 -> "21st", ...).
    """
    if 10 <= position % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position % 10, "th")
    return f"{position}{suffix}"


def _build_selection_prompt(question: str,
                            fn_defs: list[FunctionDefinition]) -> str:
    """
    Builds a prompt asking the model to pick the right function name
    for the given question, listing each available function and its
    description.
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
    Builds a prompt asking the model for a single parameter's value,
    including the function's own description and full parameter list
    as context.
    """
    param_type = func_def.parameters[param_name].type
    param_names = list(func_def.parameters.keys())
    all_params = ", ".join(param_names)
    position = param_names.index(param_name) + 1
    ordinal = _ordinal(position)
    return (f"Question: {question}\n" +
            f"The function \"{func_def.name}\" will be called: " +
            f"{func_def.description}\n" +
            "This function has the following parameters, in order: " +
            f"{all_params}\n" +
            f"What is the {ordinal} value needed for this function " +
            f"(parameter '{param_name}', a {param_type})?\n" +
            "Answer with only the value, nothing else:")


def build_parameter_prompt(model: Small_LLM_Model,
                           question: str,
                           func_def: FunctionDefinition,
                           param_name: str) -> str:
    """
    Builds the chat-wrapped prompt for a single parameter's value.
    """
    raw_prompt = _build_parameter_prompt(question, func_def, param_name)
    return _wrap_chat_prompt(model, raw_prompt)
