import inspect
from typing import Callable, Any

from google.genai import types
from pydantic import create_model

from src.agent import tools_registry


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(func).parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Tool '{func.__name__}': param '{name}' missing type annotation"
            )
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (param.annotation, default)

    Model = create_model(func.__name__, **fields)
    return Model.model_json_schema()


def tool(description: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        declaration = types.FunctionDeclaration(
            name=func.__name__,
            description=description or (func.__doc__ or "").strip(),
            parameters_json_schema=_build_parameters_schema(func),
        )
        tools_registry[func.__name__] = {
            "func": func,
            "declaration": declaration,
        }
        return func
    return decorator
