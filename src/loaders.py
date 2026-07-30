from src import FunctionDefinition, PromptEntry
import json


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """
    It parses json of function definitions to list of FunctionDefinition class.
    """
    definitions: list[FunctionDefinition]
    try:
        with open(path) as f:
            data = json.load(f)
        definitions = [FunctionDefinition(**item) for item in data]
        return definitions
    except FileNotFoundError:
        print(f"File not found at path {path}.")
    except ValueError:
        print("Failed at parsing.")


def load_prompts(path: str) -> list[PromptEntry]:
    """
    It parses json of prompts to list of PromptEntry class.
    """
    prompts: list[PromptEntry]
    try:
        with open(path) as f:
            data = json.load(f)
        prompts = [PromptEntry(**item) for item in data]
        return prompts
    except FileNotFoundError:
        print(f"File not found at path {path}.")
    except ValueError:
        print("Failed at parsing.")
