import os
import time

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(
    api_key=API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"


def make_groq_schema(schema):
    """
    Recursively make a Pydantic JSON schema compatible
    with Groq strict structured outputs.
    """

    if isinstance(schema, dict):

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        for key, value in schema.items():

            if isinstance(value, dict):
                make_groq_schema(value)

            elif isinstance(value, list):

                for item in value:

                    if isinstance(item, dict):
                        make_groq_schema(item)

    return schema


def generate_with_retry(
    max_retries=3,
    delay_seconds=4,
    **kwargs
):

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                **kwargs
            )

            return response

        except Exception as e:

            print(
                f"Retry {attempt + 1}/{max_retries} "
                f"after error: {e}"
            )

            if attempt < max_retries - 1:
                time.sleep(
                    delay_seconds * (attempt + 1)
                )

    raise RuntimeError(
        "Groq call failed after max retries"
    )