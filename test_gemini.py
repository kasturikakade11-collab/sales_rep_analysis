from agents.llm_client import client, MODEL_NAME

response = client.models.generate_content(
    model=MODEL_NAME,
    contents="Reply with exactly: Gemini connection successful."
)

print(response.text)