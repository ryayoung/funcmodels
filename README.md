# OpenAI Functions

```
pip install funcmodels
```

```py
from funcmodels import openai_function
```

# `@openai_function`

## Highlights

This documentation assumes you're already familiar with OpenAI function calling, **_and_** Pydantic BaseModel.

```py
from typing import Literal

@openai_function
def get_stock_price(ticker: str, currency: Literal["USD", "EUR"] = "USD"):
    """
    Get the stock price of a company, by ticker symbol

    Parameters
    ----------
    ticker
        The ticker symbol of the company
    currency
        The currency to use
    """
    return f"182.41 {currency}, -0.48 (0.26%) today"


get_stock_price
```
```
OpenaiFunction({
    "name": "get_stock_price",
    "description": "Get the stock price of a company, by ticker symbol",
    "parameters": {
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The ticker symbol of the company"
            },
            "currency": {
                "default": "USD",
                "enum": [
                    "USD",
                    "EUR"
                ],
                "type": "string",
                "description": "The currency to use"
            }
        },
        "required": [
            "ticker"
        ],
        "type": "object"
    }
})
```

`@openai_function` dynamically creates a custom `pydantic.BaseModel` class with:
- Your function's parameters as attributes, for validation
- Class attribute, `schema`, with an OpenAI Function object for your function
    - Parses docstring for description, and parameter descriptions, if present.
    - Type structure based on pydantic's `.model_json_schema()`
- Class method, `.from_json()` to easily instantiate your model from raw JSON arguments received from OpenAI
- A `.__call__()` method to easily call your original function, using the model's validated attributes.

---

#### Get our OpenAI function definition dictionary
```py
get_stock_price.schema
```
```
{'name': 'get_stock_price', 'description': 'Get the stock price of a company, by ticker symbol', 'parameters': {'properties': {'ticker': {'type': 'string', 'description': 'The ticker symbol of the company'}, 'currency': {'default': 'USD', 'enum': ['USD', 'EUR'], 'type': 'string', 'description': 'The currency to use'}}, 'required': ['ticker'], 'type': 'object'}}
```

#### Instantiate our pydantic model, validating arguments

```py
model = get_stock_price(ticker="AAPL")
```

#### Or, go directly from raw json arguments from OpenAI

```py
raw_arguments_from_openai = '{"ticker": "AAPL"}'
model = get_stock_price.from_json(raw_arguments_from_openai)
model.currency
```
```
'USD'
```

#### Call our function, with already-validated arguments
```py
model()
```
```
'182.41 USD, -0.48 (0.26%) today'
```

### If you prefer Pydantic syntax, we can achieve the *same* thing using `Field`s

```py
from pydantic import Field

@openai_function
def get_stock_price(
    ticker: str = Field(description="The ticker symbol of the company"),
    currency: Literal["USD", "EUR"] = Field("USD", description="The currency to use."),
):
    "Get the stock price of a company, by ticker symbol"
    return f"182.41 {currency}, -0.48 (0.26%) today"
```

Here, the field descriptions are defined in the parameters themselves, rather than the docstring.

The result is the exact same function definition as before:
```py
get_stock_price
```
```
OpenaiFunction({
    "name": "get_stock_price",
    "description": "Get the stock price of a company, by ticker symbol",
    "parameters": {
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The ticker symbol of the company"
            },
            "currency": {
                "default": "USD",
                "enum": [
                    "USD",
                    "EUR"
                ],
                "type": "string",
                "description": "The currency to use"
            }
        },
        "required": [
            "ticker"
        ],
        "type": "object"
    }
})

```
