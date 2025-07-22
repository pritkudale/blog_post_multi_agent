#!/usr/bin/env python3
"""
Simplified Tool to Get the Actual Model Name Used by the API

This module makes a direct API call to the configured endpoint
to determine the actual model name being used, as reported by the API.
This is useful when a routing service like DynaRoute is in place.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def get_actual_model_name():
    """
    Connects to the configured API endpoint and retrieves the model name
    from a test completion response.

    Returns:
        str: The actual model name reported by the API, or an error message.
    """
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    api_base_url = os.getenv('OPENAI_API_BASE')

    if not api_key or not api_base_url:
        return "Error: 'OPENAI_API_KEY' and/or 'OPENAI_API_BASE' environment variables are not set."

    print(f"Connecting to endpoint: {api_base_url}")
    print("Sending a test request to determine the model name...")

    try:
        # Initialize the OpenAI client with the custom base URL and API key
        # This client uses the same configuration your CrewAI agents would use
        client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )

        # Make a simple API call. The 'model' parameter may be required by the API,
        # but the routing service determines the actual model used.
        response = client.chat.completions.create(
            model='gpt-4o-mini',  # This can be any valid model name your endpoint expects
            messages=[{'role': 'user', 'content': 'Hi'}],
            max_tokens=2
        )

        # Access the 'model' field from the API response, as shown in your example
        actual_model = response.model

        return f"Success! The API is using the model: '{actual_model}'"

    except Exception as e:
        return f"An error occurred while contacting the API: {e}"


if __name__ == "__main__":
    model_info = get_actual_model_name()
    print("\n" + "="*60)
    print("ACTUAL MODEL NAME CHECKER")
    print("="*60)
    print(f"\n{model_info}\n")