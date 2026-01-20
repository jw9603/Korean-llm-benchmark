from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("FRIENDLI_API_KEY"),
    base_url="https://api.friendli.ai/serverless/v1",
)

completion = client.chat.completions.create(
    model="LGAI-EXAONE/K-EXAONE-236B-A23B",
    extra_body={
      "parse_reasoning": True,
      "chat_template_kwargs": {
        "enable_thinking": True
      }
    },
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "롯데이노베이트 아니?"},
    ],
)

print("Reasoning: ", completion.choices[0].message.reasoning_content)
print(completion.choices[0].message.content)