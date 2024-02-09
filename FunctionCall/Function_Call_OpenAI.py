import json
import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Define a function to simulate currency conversion.
def convert_currency(amount, from_currency="USD", to_currency="EUR"):
    """
    Simulates currency conversion from one currency to another.

    Parameters:
    - amount (float): The amount to convert.
    - from_currency (str): The currency code to convert from (default "USD").
    - to_currency (str): The currency code to convert to (default "EUR").

    Returns:
    - str: A JSON string containing the original amount, the converted amount, and the currencies involved.
    """
    # Simulated exchange rate for USD to EUR
    exchange_rate = 0.88 if from_currency == "USD" and to_currency == "EUR" else 1.14
    
    # Calculate the converted amount
    converted_amount = amount * exchange_rate
    
    # Prepare the conversion information
    conversion_info = {
        "original_amount": amount,
        "converted_amount": round(converted_amount, 2),
        "from_currency": from_currency,
        "to_currency": to_currency,
    }
    return json.dumps(conversion_info)  # Return conversion information as a JSON string


# Example usage in a conversation with OpenAI API
messages = [
    {
        "role": "user",
        "content": "How much is 100 USD in EUR?",
    }
]

# define a function
functions = [
    {
        "name": "convert_currency",
        "description": "Simulates currency conversion from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "the amount is needed to convert",
                },
                "from_currency": {"type": "string", "enum": ["USD", "EUR"]},
                 "to_currency": {"type": "string", "enum": ["USD", "EUR"]},
            },
            "required": ["amount"],
        },
    }
]


# Function call to simulate user asking about currency conversion
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": "convert_currency"},
)
#print(response)
messages.append(response["choices"][0]["message"])
args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = convert_currency(**args)

messages.append(
        {
            "role": "function",
            "name": "onvert_currency",
            "content": observation,
        }
)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
print(response)