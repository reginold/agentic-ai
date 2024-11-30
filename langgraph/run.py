import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI

client = OpenAI(
    base_url="https://api.sambanova.ai/v1",
    api_key=os.environ.get("SAMBANOVA_API_KEY")
)

completion = client.chat.completions.create(
  model="Meta-Llama-3.1-8B-Instruct",
  messages=[
      {"role": "system", "content": "Answer the question in a couple sentences."},
      {"role": "user", "content": "Share a happy story with me"}
    ]
)

result = completion.choices[0].message.content
print(result)