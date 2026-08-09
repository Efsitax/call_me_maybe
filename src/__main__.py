import src
from argparse import Namespace
from llm_sdk import Small_LLM_Model


def main() -> None:
    """
    Orchestrates the function-calling pipeline.
    """
    model = Small_LLM_Model()
    args: Namespace = src.parse_args()
    func_defs, prompts = src.load_inputs(args)
    vocab_text = src.build_vocabulary(model)
    func_names = [fn.name for fn in func_defs]
    fnl_prompt: str = src.build_selection_prompt(prompts[0].prompt, func_defs)
    chosen_function: str = src.select_candidates(model,
                                                 fnl_prompt,
                                                 func_names,
                                                 vocab_text)
    print(chosen_function)


if __name__ == "__main__":
    main()
