from __future__ import annotations
import json
import inspect
import textwrap
import docstring_parser
from typing import (
    Self,
    get_type_hints,
    Callable,
    Any,
    TypeVar,
    Generic,
    cast,
    overload,
    ClassVar,
    ParamSpec,
    TypedDict,
    Literal,
    Dict,
    Protocol,
)
from abc import ABC
from pydantic import (
    BaseModel,
    create_model,
)

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class FunctionCallResult(TypedDict):
    role: Literal["function"]
    name: str
    content: str


class OpenaiFunction(BaseModel, Generic[P, R], ABC):
    schema: ClassVar[dict[str, Any]]

    def __init__(self, *args: P.args, **kwargs: P.kwargs):
        raise NotImplementedError("OpenaiFunction is an abstract class.")

    def __call__(self) -> R:
        ...

    def execute(self) -> R:
        ...

    def execute_as_message(
        self, handle_errors: bool | Callable[[Exception], str] = False
    ) -> FunctionCallResult:
        ...

    @property
    def function(self) -> Callable[P, R]:
        ...

    @classmethod
    def from_json(cls, arguments: str) -> Self:
        ...

    @classmethod
    def from_json_to_message(
        cls, arguments: str, handle_errors: bool | Callable[[Exception], str] = False
    ) -> FunctionCallResult:
        ...

    def __getattr__(self, name: str) -> Any:
        ...


@overload
def openai_function(__func: Callable[P, R], /, **kwargs) -> type[OpenaiFunction[P, R]]:
    ...


@overload
def openai_function(
    __func: None = None, /, **kwargs
) -> Callable[[Callable[P, R]], type[OpenaiFunction[P, R]]]:
    ...


def openai_function(
    __func: None | Callable[P, R] = None, /, **kwargs
) -> (
    type[OpenaiFunction[P, R]] | Callable[[Callable[P, R]], type[OpenaiFunction[P, R]]]
):
    """
    Decorator for creating OpenAI functions.

    The returned class is created by passing your function's parameters to `pydantic.create_model()`.

    So the following code ...

        @openai_function
        def add(first: int, second: int = Field(default=0)):
            ...

    ... defines a model of this structure:

        class add(BaseModel):
            first: int
            second: int = Field(default=0)

    There are two ways the openai function differs from traditional Pydantic model behavior.
        - Parameter type hints are NOT required. The resulting model will use `Any` for params without type hints.
        - If your function can take positional arguments, then your model can also take positional arguments.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to `pydantic.create_model()`, before the attributes.
    """

    def decorator(fn):
        function_name = fn.__name__

        # Param names and default values obtained from `inspect.signature()`.
        # `get_type_hints()` MUST be used for annotations since it evaluates forward references.

        inspected_params = list(inspect.signature(fn).parameters.values())
        param_names = [p.name for p in inspected_params]
        param_types = get_type_hints(fn)

        blacklisted_param_names = {"schema", "function", "execute", "from_json"}
        for name in blacklisted_param_names:
            if name in param_names:
                raise ValueError(
                    f"Parameter name '{name}' is reserved for OpenAI functions."
                )

        param_defaults = {
            param.name: param.default if param.default is not param.empty else ...
            for param in inspected_params
        }

        model_fields: dict[str, Any] = {
            name: (param_types.get(name, Any), param_defaults[name])
            for name in param_names
        }

        # Passing model fields last, so kwargs can't interfere
        Base = create_model(function_name, **{**kwargs, **model_fields})
        function_schema = get_schema(fn, Base)

        # This might seem a bit 'extra', but it's a great convenience to see your resulting
        # function schema neatly printed, as this is something you'll certainly want to do.
        class BaseMeta(type(Base)):
            def __repr__(self):
                return f"OpenaiFunction({json.dumps(function_schema, indent=4)})"

        class Model(Base, metaclass=BaseMeta):
            schema: ClassVar[dict[str, object]] = function_schema

            def __init__(self, *args, **kwargs):
                # Pydantic doesn't allow positionals, so we move args to kwargs.
                for i, arg in enumerate(args):
                    name = param_names[i]
                    if name not in kwargs:
                        kwargs[name] = arg

                super().__init__(**kwargs)

            def execute(self):
                ...  # Implemented conditionally, below

            def execute_as_message(
                self, handle_errors: bool | Callable[[Exception], str] = False
            ):
                return execute_function_as_message(
                    self.execute, function_name, handle_errors
                )

            def __call__(self):
                return self.execute()

            @property
            def function(self):
                return fn

            @classmethod
            def from_json(cls, arguments: str):
                parsed = json.loads(arguments)
                return cls(**parsed)

            @classmethod
            def from_json_to_message(
                cls,
                arguments: str,
                handle_errors: bool | Callable[[Exception], str] = False,
            ):
                return execute_json_args_as_message(
                    cls, function_name, arguments, handle_errors
                )

        # If the function has positional-only params, `__call__` should handle them.
        # Notice we aren't adjusting `__init__`, because the whole point of this class is
        # to support arguments coming from OpenAI (JSON) which will be keyword arguments anyway.
        positional_only_params = [
            p.name
            for p in inspected_params
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
        ]
        if positional_only_params:

            def execute(self):
                kwargs = self.model_dump()
                args = [kwargs.pop(name) for name in positional_only_params]
                return fn(*args, **kwargs)

        else:

            def execute(self):
                return fn(**self.model_dump())

        Model.execute = execute
        Model.__name__ = function_name
        result = cast(type[OpenaiFunction[P, R]], Model)
        return result

    if __func is None:
        return decorator

    return decorator(__func)


def get_schema(fn: Callable, model: type[BaseModel]) -> dict[str, object]:
    schema: dict[str, object] = {
        "name": fn.__name__,
    }
    parameters = model.model_json_schema()
    parameters.pop("title")
    parameters = remove_key_recursive(parameters, "title")
    schema["parameters"] = parameters

    raw_docstring = fn.__doc__
    if not raw_docstring:
        return schema

    docstring = textwrap.dedent(raw_docstring).strip()

    doc = docstring_parser.parse(docstring)

    if doc.params:
        descriptions = {
            param.arg_name: param.description
            for param in doc.params
            if param.description
            and param.arg_name
            and param.arg_name in parameters["properties"]
        }
        for name, description in descriptions.items():
            schema["parameters"]["properties"][name]["description"] = description

    docstring_without_params = remove_parameters_section_from_docstring(
        docstring
    ).strip()
    if not docstring_without_params:
        return schema

    schema["description"] = docstring_without_params

    # Re-order keys so description is first. Better readability.
    return {
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["parameters"],
    }


def remove_key_recursive(item, key_to_remove) -> Any:
    if isinstance(item, dict):
        return {
            key: remove_key_recursive(value, key_to_remove)
            for key, value in item.items()
            if key != key_to_remove
        }
    elif isinstance(item, (list, tuple)):
        return [remove_key_recursive(value, key_to_remove) for value in item]
    else:
        return item


def remove_parameters_section_from_docstring(docstring: str) -> str:
    numpydoc_sections = {
        "Parameters",
        "Returns",
        "Examples",
        "Raises",
        "Notes",
        "References",
        "Yields",
    }
    lines = docstring.split("\n")
    params_start = None
    params_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip() == "Parameters":
            params_start = i
            continue

        if params_start != -1 and any(
            line.strip() == section for section in numpydoc_sections
        ):
            params_end = i
            break

    if params_start is None:
        return docstring

    new_lines = lines[:params_start] + lines[params_end:]
    return "\n".join(new_lines)


def execute_function_as_message(
    function: Callable,
    name: str,
    handle_errors: bool | Callable[[Exception], str] = False,
) -> FunctionCallResult:
    if not handle_errors:
        return {
            "role": "function",
            "name": name,
            "content": str(function()),
        }
    handler = (
        handle_errors
        if callable(handle_errors)
        else (lambda e: f"{type(e).__name__}: {e}")
    )
    try:
        return {
            "role": "function",
            "name": name,
            "content": str(function()),
        }
    except Exception as e:
        return {
            "role": "function",
            "name": name,
            "content": handler(e),
        }


def execute_json_args_as_message(
    openai_func: OpenaiFunction,
    function_name: str,
    arguments: str,
    handle_errors: bool | Callable[[Exception], str] = False,
) -> FunctionCallResult:
    if not handle_errors:
        parsed = json.loads(arguments)
        return openai_func(**parsed).execute_as_message()

    handler = (
        handle_errors
        if callable(handle_errors)
        else (lambda e: f"{type(e).__name__}: {e}")
    )
    try:
        parsed = json.loads(arguments)
        return openai_func(**parsed).execute_as_message()
    except Exception as e:
        return {
            "role": "function",
            "name": function_name,
            "content": handler(e),
        }


class OpenaiFunctionGroup(dict[str, type[OpenaiFunction]]):
    def __init__(self):
        return dict.__init__({})

    @property
    def functions(self) -> list[type[OpenaiFunction]]:
        return list(self.values())

    @property
    def function_definitions(self) -> list[dict[str, object]]:
        return [func.schema for func in self.values()]

    def add(self, function: Callable | type[OpenaiFunction]):
        if not isinstance(function, type(BaseModel)):
            function = openai_function(function)
        key = function.schema["name"]
        self[key] = cast(type[OpenaiFunction], function)

    def get_valid_function_call(
        self, function_call: dict[str, str] | OpenaiV0FuncCall | OpenaiV1FuncCall
    ) -> OpenaiFunction:
        name, arguments = unpack_func_call(function_call)
        return self[name].from_json(arguments)

    def function_call_to_message(
        self,
        function_call: dict[str, str] | OpenaiV0FuncCall | OpenaiV1FuncCall,
        handle_errors: bool | Callable[[Exception], str] = False,
    ) -> FunctionCallResult:
        name, arguments = unpack_func_call(function_call)
        return self[name].from_json_to_message(arguments, handle_errors)

    def __repr__(self):
        return f"OpenaiFunctionGroup({json.dumps(self.function_definitions, indent=4)})"


class OpenaiV0FuncCall(TypedDict):
    name: str
    arguments: str


class OpenaiV1FuncCall(Protocol):
    name: str
    arguments: str


def unpack_func_call(
    function_call: dict[str, str] | OpenaiV0FuncCall | OpenaiV1FuncCall
) -> tuple[str, str]:
    if isinstance(function_call, dict):
        return function_call["name"], function_call["arguments"]
    return function_call.name, function_call.arguments


def openai_function_group(
    functions: list[Callable | type[OpenaiFunction]] | None = None,
) -> OpenaiFunctionGroup:
    group = OpenaiFunctionGroup()

    for func in functions or []:
        group.add(func)

    return group
