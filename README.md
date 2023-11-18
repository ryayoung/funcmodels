# OpenAI Functions

```
pip install funcmodels
```

```py
from funcmodels import openai_function
```

# `@openai_function`

The most intuitive, robust and "pure" way to implement functions for OpenAI function calling.

Designed as a more straightforward and ergonomic alternative to [jxnl/instructor](https://github.com/jxnl/instructor). Rather than defining your function as a `BaseModel`, you define your function as a **_function_**, and the `BaseModel` is created for you.

With `@openai_function`, the ...
- **Pydantic** model for data validation,
- definition of your **OpenAI Function** JSON schema,
- parsing and validation of raw json string arguments,
- logic for handling the arguments and producing a result

... are all **explicitly defined and encapsulated in one place**: your decorated function.

`@openai_function` will turn your function into a Pydantic BaseModel class, using your function parameters as attributes.
It parses your docstring for a description and parameter descriptions, and combines those with the BaseModel's JSON
schema to produce a complete OpenAI Function definition. You'll receive back an extended version of this BaseModel class,
equipped with a class attribute, `.schema` with the function definition, a `.from_json()` class method for creation and validation
directly from raw function call arguments from OpenAI, and an `execute()` method that passes the instance's validated attributes to the
handler defined by your decorated function.

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

#### Get our OpenAI function definition dictionary
```py
get_stock_price.schema
```
```
{'name': 'get_stock_price', 'description': 'Get the stock price of a company, by ticker symbol', 'parameters': {'properties': {'ticker': {'type': 'string', 'description': 'The ticker symbol of the company'}, 'currency': {'default': 'USD', 'enum': ['USD', 'EUR'], 'type': 'string', 'description': 'The currency to use'}}, 'required': ['ticker'], 'type': 'object'}}
```

#### Instantiate our pydantic model, validating arguments

```py
validated_function_call = get_stock_price(ticker="AAPL")
```

#### Or, go directly from raw json arguments from OpenAI

```py
raw_arguments_from_openai = '{"ticker": "AAPL"}'
validated_function_call = get_stock_price.from_json(raw_arguments_from_openai)
validated_function_call.currency
```
```
'USD'
```

#### Call our function, with already-validated arguments
```py
validated_function_call.execute()
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

# Function Groups

```py
def get_stock_price(ticker: str):
    return '182.41 USD, -0.48 (0.26%) today'

def get_weather(city: str):
    return "Sunny, 72 degrees, 0% chance of rain"

group = openai_function_group(functions=[get_stock_price, get_weather])
group
```
```
OpenaiFunctionGroup([
    {
        "name": "get_stock_price",
        "parameters": {
            "properties": {
                "ticker": {
                    "type": "string"
                }
            },
            "required": [
                "ticker"
            ],
            "type": "object"
        }
    },
    {
        "name": "get_weather",
        "parameters": {
            "properties": {
                "city": {
                    "type": "string"
                }
            },
            "required": [
                "city"
            ],
            "type": "object"
        }
    }
])
```

> Note: Use `group.function_definitions` to get a list of the raw function schemas to use in the `functions` argument to OpenAI

#### Now, when we get a function call from OpenAI, we can let the group handle it.
```py
function_call_from_openai = {
    "name": "get_weather",
    "arguments": '{"city": "Denver"}',
}

validated_function_call = group.evaluate_function_call(function_call_from_openai)
validated_function_call
```
```
get_weather(city='Denver')
```

```py
result = validated_function_call.execute()
result
```
```
'Sunny, 72 degrees, 0% chance of rain'
```


# Create your own ChatGPT

Because each `@openai_function` encapsulates all the information needed to facilitate OpenAI function calling, it's
very easy for us to build our own ChatGPT, with **automated function call handling**.

## Step 1. Define conversation handler

```py
import json
from dataclasses import dataclass
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
from funcmodels import OpenaiFunctionGroup, openai_function_group


@dataclass
class ChatGPTConversation:
    model: str
    client: OpenAI
    functions: OpenaiFunctionGroup
    messages: list[dict]

    def get_openai_response(self) -> Choice:
        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            functions=self.functions.function_definitions,
            function_call="auto",
        )
        return response.choices[0]

    def handle_function_call(function_call) -> dict:
        # match the function call to the right function, and validate arguments
        validated_call = self.functions.evaluate_function_call(function_call)
        # Call the function with the validated arguments
        result = validated_call.execute()
        return dict(role="function", name=function_call.name, content=str(result))

    def chat(self):
        result = self.get_openai_response()
        self.add_message(result.message.model_dump(exclude_unset=True))

        if (function_call := result.message.function_call) is not None:
            self.add_message(self.handle_function_call(function_call))
            if result.finish_reason == 'function_call':
                self.chat()

    def add_message(self, message: dict):
        print(json.dumps(message, indent=4))
        self.messages.append(message)

    def send_message(self, prompt: str):
        self.add_message({"role": "user", "content": prompt})
        self.chat()
```

> Here, `chat()` is a recursive function that continues sending API requests
for as long as the response's `finish_reason='function_call'`.

## Step 2. Define openai functions
```py
def get_stock_price(ticker: str):
    "Get the stock price of a company, by ticker symbol."
    return "182.41 USD, −0.48 (0.26%) today"

def get_weather(city: str):
    "Get the current weather in a city."
    return "Sunny and 75 degrees"

def get_current_datetime(city: str):
    "Get the current date and time in a city."
    return "Friday, Nov. 10, 2023, 10:00 AM"

group = openai_function_group(functions=[get_stock_price, get_weather, get_current_datetime])
group
```
```
OpenaiFunctionGroup([
    {
        "name": "get_stock_price",
        "description": "Get the stock price of a company, by ticker symbol.",
        "parameters": {
            "properties": {
                "ticker": {
                    "type": "string"
                }
            },
            "required": [
                "ticker"
            ],
            "type": "object"
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "properties": {
                "city": {
                    "type": "string"
                }
            },
            "required": [
                "city"
            ],
            "type": "object"
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time in a city.",
        "parameters": {
            "properties": {
                "city": {
                    "type": "string"
                }
            },
            "required": [
                "city"
            ],
            "type": "object"
        }
    }
])
```

## Step 3. Create a new conversation
```py
chatgpt = ChatGPTConversation(
    model="gpt-4-1106-preview",
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    functions=group,
    messages=[
        dict(role="system", content="You are a helpful AI assistant."),
    ]
)
```

## Step 4. Exchange messages

**You'll need to use Jupyter notebooks (or interactive terminal) for this**

```py
chatgpt.send_message("Hello, how are you?")
```
```
{
    "role": "user",
    "content": "Hello, how are you?"
}
{
    "content": "Hello! I'm just a machine, so I don't have feelings, but I'm functioning optimally. How can I assist you today?",
    "role": "assistant"
}
```

```py
chatgpt.send_message(
    "I'm enjoying my breakfast here in Denver. Can you list 3 fun things to do here, "
    "then give me a quick morning update with the information available?"
)
```
```
{
    "role": "user",
    "content": "I'm enjoying my breakfast here in Denver. Can you list 3 fun things to do here, then give me a quick morning update with the information available?"
}
{
    "content": "Of course! Here are three fun activities to do in Denver:\n\n1. **Visit the Denver Botanic Gardens**: Enjoy a peaceful morning walking through the various plant collections and exhibits. It's a good way to appreciate nature and the diverse plant life.\n\n2. **Explore the Denver Art Museum**: With its wide range of art from across the world, including contemporary, indigenous, and classic pieces, this museum offers a rich cultural experience.\n\n3. **Take a Stroll Around Larimer Square**: This historic district is perfect for a leisurely walk, as you can explore boutique shops, enjoy local eateries, and take in the unique architecture and vibrant atmosphere.\n\nNow, let's get you that morning update for Denver:\n\nI can provide you with the current weather, the current date and time, and check on the stock price of a company if you're following any in particular. Let's start with the weather and the date and time:\n\n- **Weather**: I'll check the current weather conditions for you.\n- **Current Date and Time**: I\u2019ll provide you with the current date and time in Denver.\n\nLet me fetch the latest information for you, just a moment.",
    "role": "assistant",
    "function_call": {
        "arguments": "{\"city\":\"Denver\"}",
        "name": "get_weather"
    }
}
{
    "role": "function",
    "name": "get_weather",
    "content": "Sunny and 75 degrees"
}
{
    "content": null,
    "role": "assistant",
    "function_call": {
        "arguments": "{\"city\":\"Denver\"}",
        "name": "get_current_datetime"
    }
}
{
    "role": "function",
    "name": "get_current_datetime",
    "content": "Friday, Nov. 10, 2023, 10:00 AM"
}
{
    "content": "Here's your morning update for Denver:\n\n- **Weather:** It's currently sunny and 75 degrees in Denver. It seems like a beautiful day to enjoy any of the outdoor activities I mentioned!\n- **Current Date and Time:** It's Friday, November 10, 2023, and the time is 10:00 AM.\n\nIf there's a particular stock you'd like to check on, just let me know the ticker symbol, and I'll get the latest price for you. Have a fantastic day enjoying Denver!",
    "role": "assistant"
}
```

#### If we stitch together the `content` of each *assistant* message (2nd and last messages), we get a continuous block of response text:

> Denver is a vibrant city with plenty of activities to enjoy, ranging from cultural experiences to outdoor adventures. Here are three fun things you might consider doing in Denver:
> 
> 1. **Visit Denver Botanic Gardens**: These gardens offer a peaceful and beautiful urban retreat. You can stroll through various plant displays, including a Japanese garden and a conservatory with exotic tropical and subtropical species.
> 
> 2. **Explore the Denver Art Museum**: Known for its collection of American Indian Art, the Denver Art Museum also boasts a wide range of other collections, from pre-Columbian artifacts to contemporary art. The museum's architecture is also a work of art in itself.
> 
> 3. **Discover Red Rocks Park and Amphitheatre**: Just outside Denver, this world-famous outdoor venue is renowned for its amazing acoustics and stunning red sandstone formations. During the day, you can hike or bike the trails, and in the evening, possibly catch a concert under the stars.
> 
> Now, let's get you a morning update on Denver:
> 
> - **Weather**: I'll check the current weather conditions for you.
> - **Local Time**: I'll provide the current date and time.
> - **Stock Market**: If you're interested in a particular stock, I can provide the latest price.
> 
> Let me gather the weather and local time information for you. One moment, please.
> Here's your morning update for Denver:
> 
> - **Weather**: It's a beautiful sunny day with a pleasant temperature of 75 degrees Fahrenheit. A great day to plan an outdoor activity!
> - **Local Time**: The current date and time in Denver is Friday, November 10th, 2023, at 10:00 AM.
> 
> If you have any stock in mind or need more information, feel free to let me know! Enjoy your breakfast and have a fantastic day in Denver.


#### This single response was made up of multiple API calls/responses:
1. *Sent*:
    - User prompt
2. *Received*: 
    - Content response (**PART 1**)
    - Function call to: `get_weather`
3. *Sent*:
    - Function response from: `get_weather`
4. *Received*:
    - Function call to: `get_current_datetime`
5. *Sent*:
    - Function response from: `get_current_datetime`
6. *Received*:
    - Content response (**PART 2**)

The response text above combines the content from API responses **2** and **6**.
