import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Testing if Groq rate limit reset...")

try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hi!"}],
        max_tokens=20
    )
    print(f"✅ Groq WORKS! Rate limit has reset!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Still rate limited: {e}")