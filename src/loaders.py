from src import FunctionDefinition, PromptEntry, FunctionCallResult
import json
import sys
from pydantic import ValidationError
from argparse import Namespace
from pathlib import Path


def _load_function_definitions(path: str) -> list[FunctionDefinition]:
    """
    It parses json of function definitions to list of FunctionDefinition class.
    """
    definitions: list[FunctionDefinition]
    with open(path) as f:
        data = json.load(f)
    definitions = [FunctionDefinition(**item) for item in data]
    return definitions


def _load_prompts(path: str) -> list[PromptEntry]:
    """
    It parses json of prompts to list of PromptEntry class.
    """
    prompts: list[PromptEntry]
    with open(path) as f:
        data = json.load(f)
    prompts = [PromptEntry(**item) for item in data]
    return prompts


def load_inputs(args: Namespace) -> tuple[list[FunctionDefinition],
                                          list[PromptEntry]]:
    try:
        func_defs: list[FunctionDefinition]
        func_defs = _load_function_definitions(
            args.functions_definition
        )
        prompts: list[PromptEntry] = _load_prompts(args.input)
        return (func_defs, prompts)
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Failed at parsing on JSON.")
        sys.exit(1)
    except ValidationError:
        print("Validation Error.")
        sys.exit(1)


def _write_results(results: list[FunctionCallResult], path: str) -> None:
    """
    Writes results as a JSON array to path, creating parent
    directories as needed.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_results(results: list[FunctionCallResult], path: str) -> None:
    """
    Writes results to path, exiting with a clear message on failure.
    """
    try:
        _write_results(results, path)
    except (OSError, TypeError) as e:
        print(f"Could not write output file: {e}")
        sys.exit(1)
