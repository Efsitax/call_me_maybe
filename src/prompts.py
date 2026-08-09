from src import FunctionDefinition


def build_selection_prompt(question: str,
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
