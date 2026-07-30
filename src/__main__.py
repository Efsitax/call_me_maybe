from src import FunctionDefinition, PromptEntry
from src import parse_args, load_prompts, load_function_definitions
from argparse import Namespace


def main() -> None:
    """
    Orchestrates the function-calling pipeline.
    """
    args: Namespace = parse_args()
    func_defs: list[FunctionDefinition] = load_function_definitions(
        args.functions_definition
    )
    prompts: list[PromptEntry] = load_prompts(args.input)
    print(func_defs)
    print("************************************")
    print(prompts)
    print("************************************")
    print(args.output)


if __name__ == "__main__":
    main()
