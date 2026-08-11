import src
from argparse import Namespace
from llm_sdk import Small_LLM_Model
from typing import Any


def _resolve_parameters(model: Small_LLM_Model,
                        question: str,
                        func_def: src.FunctionDefinition,
                        vocab_text: list[str | None],
                        stop_token_id: int) -> dict[str, Any]:
    """
    Resolves the value of every parameter of func_def for the question.
    """
    parameters: dict[str, Any] = {}
    for param_name, param in func_def.parameters.items():
        prompt = src.build_parameter_prompt(model, question, func_def,
                                            param_name)
        if param.type == "number":
            used_numbers = {v for v in parameters.values()
                            if isinstance(v, float)}
            parameters[param_name] = src.generate_number(
                model, prompt, vocab_text, stop_token_id,
                forbidden_values=used_numbers
            )
        elif param.type == "string":
            parameters[param_name] = src.generate_string(
                model, prompt, vocab_text, stop_token_id
            )
        else:
            raise NotImplementedError(
                f"Parameter type {param.type!r} not supported yet."
            )
    return parameters


def main() -> None:
    """
    Orchestrates the function-calling pipeline.
    """
    model = Small_LLM_Model()
    args: Namespace = src.parse_args()
    func_defs, prompts = src.load_inputs(args)
    vocab_text = src.build_vocabulary(model)
    stop_token_id: int = src.find_special_token_id(model, "<|im_end|>")
    func_names = [fn.name for fn in func_defs]
    for entry in prompts:
        try:
            fn_prompt: str = src.build_function_prompt(model, entry.prompt,
                                                       func_defs)
            chosen_function: str = src.select_candidates(model, fn_prompt,
                                                         func_names,
                                                         vocab_text)
            func_def: src.FunctionDefinition = next(
                fn for fn in func_defs if fn.name == chosen_function
            )
            parameters = _resolve_parameters(model, entry.prompt, func_def,
                                             vocab_text, stop_token_id)
            print(f"{entry.prompt} -> {chosen_function} " +
                  f"parameters: {parameters}")
        except Exception as e:
            print(f"Error Occured: {e}")


if __name__ == "__main__":
    main()
