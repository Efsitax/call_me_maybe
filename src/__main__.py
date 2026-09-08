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
        if param.type == "number" or param.type == "integer":
            used_numbers = {v for v in parameters.values()
                            if isinstance(v, (float, int))}
            parameters[param_name] = src.generate_number(
                model, prompt, param.type, vocab_text, stop_token_id,
                question=question, forbidden_values=used_numbers
            )
        elif param.type == "string":
            parameters[param_name] = src.generate_string(
                model, prompt, vocab_text, stop_token_id,
                question=question
            )
        elif param.type == "boolean":
            result = src.select_candidates(
                model, prompt, ["true", "false"], vocab_text
            )
            parameters[param_name] = result == "true"
        else:
            raise NotImplementedError(
                f"Parameter type {param.type!r} not supported yet."
            )
    return parameters


def main() -> None:
    """
    Orchestrates the function-calling pipeline.
    """
    args: Namespace = src.parse_args()
    model = Small_LLM_Model(args.model)
    func_defs, prompts = src.load_inputs(args)
    vocab_text = src.build_vocabulary(model)
    stop_token_id: int = src.find_stop_token_id(model)
    func_names = [fn.name for fn in func_defs]
    call_results: list[src.FunctionCallResult] = []
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
            parameters: dict[str, Any] = _resolve_parameters(model,
                                                             entry.prompt,
                                                             func_def,
                                                             vocab_text,
                                                             stop_token_id)
            print(f"{entry.prompt} -> {chosen_function} " +
                  f"parameters: {parameters}")
            call_result = src.FunctionCallResult(
                prompt=entry.prompt,
                name=chosen_function,
                parameters=parameters
            )
            call_results.append(call_result)
        except Exception as e:
            print(f"Error Occured: {e}")
            call_result = src.FunctionCallResult(
                prompt=entry.prompt,
                name="",
                parameters={}
            )
            call_results.append(call_result)
    src.save_results(call_results, args.output)


if __name__ == "__main__":
    main()
