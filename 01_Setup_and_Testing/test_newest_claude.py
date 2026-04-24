import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Try the absolute newest model names
newest_models = [
    "claude-sonnet-4-20250514",  # Newest Sonnet
    "claude-opus-4-20250514",    # Newest Opus
    "claude-3-5-sonnet-latest",  # Latest 3.5
]

for model in newest_models:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✅ {model} WORKS!")
        break
    except Exception as e:
        print(f"❌ {model}: {str(e)[:80]}")