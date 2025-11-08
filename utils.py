def get_provider(model_name: str) -> str:
    """Determine the provider for a given model"""
    if any(model_name.startswith(prefix) for prefix in ["gpt-", "openai/"]):
        return "OpenAI"
    return "Groq"
