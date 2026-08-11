import src
from argparse import Namespace
from llm_sdk import Small_LLM_Model
from typing import Any


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
            fnl_prompt_func: str = src.build_function_prompt(model,
                                                             entry.prompt,
                                                             func_defs)
            chosen_function: str = src.select_candidates(model,
                                                         fnl_prompt_func,
                                                         func_names,
                                                         vocab_text)
            func_def: src.FunctionDefinition = next(
                fn for fn in func_defs if fn.name == chosen_function
            )
            parameters: dict[str, Any] = {}
            for param_name, param_type in func_def.parameters.items():
                fnl_prompt_param: str = src.build_parameter_prompt(
                    model,
                    entry.prompt,
                    func_def,
                    param_name
                )
                if param_type.type == "number":
                    used_numbers = {
                        v for v in parameters.values()
                        if isinstance(v, float)
                    }
                    value = src.generate_number(model,
                                                fnl_prompt_param,
                                                vocab_text,
                                                stop_token_id,
                                                forbidden_values=used_numbers)
                    parameters[param_name] = value
                else:
                    raise NotImplementedError(
                        f"Parameter type {param_type.type!r} " +
                        "not supported yet.")
            print(f"{entry.prompt} -> {chosen_function} " +
                  f"parameters: {parameters}")
        except Exception as e:
            print(f"Error Occured: {e}")


if __name__ == "__main__":
    main()
