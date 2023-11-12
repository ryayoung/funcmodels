import inspect
import textwrap
import docstring_parser
from typing import (
    get_type_hints,
    Callable,
    Any,
    TypeVar,
    Generic,
    cast,
    overload,
    ClassVar,
)
from pydantic import (
    BaseModel,
    create_model,
)

ReturnType = TypeVar("ReturnType", covariant=True)


class CallableModel(BaseModel, Generic[ReturnType]):
    function_schema: ClassVar[dict[str, object]]

    def __init__(self, **_: Any):
        ...

    def __call__(self) -> ReturnType:
        ...


@overload
def openai_function(
    __func: Callable[..., ReturnType], /,
) -> type[CallableModel[ReturnType]]:
    ...


@overload
def openai_function(
    __func: None = None, /, 
) -> Callable[[Callable[..., ReturnType]], type[CallableModel[ReturnType]]]:
    ...


def openai_function(
    __func: None | Callable[..., ReturnType] = None, /,
) -> (
    type[CallableModel[ReturnType]]
    | Callable[[Callable[..., ReturnType]], type[CallableModel[ReturnType]]]
):
    def decorator(fn):
        # Get annotations from get_type_hints() since it evaluates forward references.
        # Get defaults and names from inspect.signature() since it includes all of them.
        # Replace empty defaults with `...` since that's what Pydantic expects.

        param_types = get_type_hints(fn)

        def coalesce_empty_with_ellipsis(param: inspect.Parameter) -> Any:
            return param.default if param.default is not param.empty else ...

        param_defaults = {
            name: coalesce_empty_with_ellipsis(param)
            for name, param in inspect.signature(fn).parameters.items()
        }

        param_names = list(param_defaults)
        assert 'openai_function' not in param_names, "openai_function is a reserved parameter name"

        model_fields: dict[str, Any] = {
            name: (param_types.get(name, Any), param_defaults[name]) for name in param_names
        }

        base_model = create_model(fn.__name__ + "Model", **model_fields)

        def __call__(self):
            # call original func with model's attrs
            return fn(**self.model_dump())

        # Need to use `cast` here, since we're adding a new method after class creation
        callable_model = cast(type[CallableModel[ReturnType]], base_model)
        callable_model.__call__ = __call__
        callable_model.function_schema = get_schema(fn, base_model)
        return callable_model

    if __func is None:
        return decorator  # type: ignore

    return decorator(__func)


def get_schema(fn: Callable, model: type[BaseModel]) -> dict[str, object]:
    schema: dict[str, object] = {
        "name": fn.__name__,
    }
    parameters = model.model_json_schema()
    parameters.pop('title')
    parameters = remove_key_recursive(parameters, 'title')
    parameters = remove_key_value_recursive(parameters, 'default', None)
    schema["parameters"] = parameters

    raw_docstring = fn.__doc__
    if not raw_docstring:
        return schema

    doc = docstring_parser.parse(textwrap.dedent(raw_docstring))
    include_in_schema_description = [
        'short_description',
        'long_description',
        'examples',
        'deprecation',
        'raises',
    ]
    values = [
        getattr(doc, attr, None) for attr in include_in_schema_description
    ]
    if doc.returns:
        parts = [
            doc.returns.return_name,
            doc.returns.type_name,
            doc.returns.description,
        ]
        parts = [p for p in parts if p]
        if parts:
            values.append(" - ".join(parts))

    filtered_values = [v for v in values if v]
    schema["description"] = "\n\n".join(filtered_values)

    if doc.params:
        descriptions = {
            param.arg_name: param.description for param in doc.params
            if param.description and param.arg_name
            and param.arg_name in parameters['properties']
        }
        for name, description in descriptions.items():
            schema['parameters']['properties'][name]['description'] = description

    return schema


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


def remove_key_value_recursive(item, key_to_remove, value_to_remove) -> Any:
    "Remove key/value pairs from a dict"
    if isinstance(item, dict):
        return {
            key: remove_key_value_recursive(value, key_to_remove, value_to_remove)
            for key, value in item.items()
            if not (key == key_to_remove and value == value_to_remove)
        }
    elif isinstance(item, (list, tuple)):
        return [remove_key_value_recursive(value, key_to_remove, value_to_remove) for value in item]
    else:
        return item


if __name__ == "__main__":

    @openai_function
    def get_stock_price(ticker: str):
        """
        Get the stock price of a company, by ticker symbol
        """
        return "182.41 USD, -0.48 (0.26%) today"

    from dictkit.render import render
    print(render(get_stock_price.function_schema))

    validated = get_stock_price(ticker="AAPL")
    result = validated()
    print(result)
