import functools
import inspect
from typing import (
    get_type_hints,
    Callable,
    Any,
    TypeVar,
    Generic,
    cast,
    overload,
    Literal,
)
from pydantic import (
    BaseModel,
    create_model,
    Field,
)

ReturnType = TypeVar("ReturnType", covariant=True)


class CallableModel(BaseModel, Generic[ReturnType]):
    def __init__(self, **_: Any):
        ...

    def __call__(self) -> ReturnType:
        ...


@overload
def funcmodel(
    __func: Callable[..., ReturnType],
    /,
    *,
    call: Literal[True],
) -> Callable[..., ReturnType]:
    ...


@overload
def funcmodel(
    __func: Callable[..., ReturnType],
    /,
    *,
    call: Literal[False] = False,
) -> type[CallableModel[ReturnType]]:
    ...


@overload
def funcmodel(
    __func: None = None, 
    /, 
    *, 
    call: Literal[True],
) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    ...


@overload
def funcmodel(
    __func: None = None, 
    /, 
    *, 
    call: Literal[False],
) -> Callable[[Callable[..., ReturnType]], type[CallableModel[ReturnType]]]:
    ...


def funcmodel(
    __func: None | Callable[..., ReturnType] = None,
    /,
    *,
    call: bool = False,
) -> (
    Callable[..., ReturnType]
    | type[CallableModel[ReturnType]]
    | Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]
    | Callable[[Callable[..., ReturnType]], type[CallableModel[ReturnType]]]
):
    def decorator(fn):
        # using get_type_hints() instead of inspect.signature() bcas it evaluates forward references
        param_types = get_type_hints(fn)

        # Pydantic expects `...` when a default value is not provided.
        def coalesce_empty_with_ellipsis(param: inspect.Parameter) -> Any:
            return param.default if param.default is not param.empty else ...

        # using `inspect.signature` bcas it provides default values
        param_defaults = {
            name: coalesce_empty_with_ellipsis(param)
            for name, param in inspect.signature(fn).parameters.items()
        }

        # use names from `inspect.signature` params because it includes all of them, including those not annotated
        param_names = list(param_defaults)

        model_fields: dict[str, Any] = {
            name: (param_types[name], param_defaults[name]) for name in param_names
        }

        base_model = create_model(fn.__name__ + "Model", **model_fields)

        if call:
            # wrap original func, to mimic its functionality
            @functools.wraps(fn)
            def wrapper(*args, **kwargs) -> ReturnType:
                for i, arg in enumerate(args):
                    kwargs[param_names[i]] = arg
                model_instance = base_model(**kwargs)
                return fn(**model_instance.model_dump())

            return wrapper

        def __call__(self):
            # call original func with model's attrs
            return fn(**self.model_dump())

        # Need to use `cast` here, since we're adding a new method after class creation
        callable_model = cast(type[CallableModel[ReturnType]], base_model)
        callable_model.__call__ = __call__
        return callable_model

    if __func is None:
        return decorator  # type: ignore

    return decorator(__func)


if __name__ == "__main__":

    @funcmodel
    def func_validated(
        a: str, b: list[str | int], c: list[int] = Field(default_factory=list)
    ) -> int:
        print(a, b, c)
        return 5

    validated_callable_model = func_validated(a="hi", b=['a', 5])
    result = validated_callable_model()
    print(round(result, 2))


    @funcmodel(call=True)
    def func_auto_validated(
        a: str, b: list[str | int], c: list[int] = Field(default_factory=list)
    ) -> int:
        print(a, b, c)
        return 5

    result = func_auto_validated(a="hi", b=[])
    print(round(result, 2))
