from pydantic import BaseModel


class Parameter(BaseModel):
    """
    Represents a function parameter's JSON schema type.
    """
    type: str


class ReturnType(BaseModel):
    """
    Represents a function's return JSON schema type.
    """
    type: str


class FunctionDefinition(BaseModel):
    """
    Represents a single function definition from functions_definition.json.
    """
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class PromptEntry(BaseModel):
    """
    Represents a single prompt entry from function_calling_tests.json.
    """
    prompt: str
