# PLAI OpenAI Compatible API

Endpoints:
- GET /v1/models
- POST /v1/chat/completions

Supports:
- Streaming
- Auto cookie refresh
- Turnstile solving
- OpenAI-compatible API

Once your API version is deployed on Render, your base URL will look like:

```text
https://your-app-name.onrender.com
```

Then use OpenAI-compatible endpoints.

# List models

```bash
curl https://your-app-name.onrender.com/v1/models
```

---

# Chat completion (non-streaming)

```bash
curl https://your-app-name.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nano",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing"
      }
    ]
  }'
```

---

# Streaming response

```bash
curl https://your-app-name.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "nano",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Generate 500 prompts"
      }
    ]
  }'
```

`-N` is important for realtime streaming.

---

# Supported models

Short aliases:

```text
nano
super
vision
```

Or full names:

```text
nvidia/nemotron-3-nano-30b-a3b:free
nvidia/nemotron-3-super-120b-a12b:free
nvidia/nemotron-nano-12b-v2-vl:free
```

---

# Python OpenAI SDK

Install:

```bash
pip install openai
```

Use:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-app-name.onrender.com/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="nano",
    messages=[
        {
            "role": "user",
            "content": "Hello"
        }
    ]
)

print(response.choices[0].message.content)
```

---

# Python streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-app-name.onrender.com/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="nano",
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "Generate 1000 prompts"
        }
    ]
)

for chunk in stream:

    if chunk.choices:

        delta = chunk.choices[0].delta.content

        if delta:
            print(delta, end="", flush=True)
```

---

# JavaScript

```javascript
const response = await fetch(
  "https://your-app-name.onrender.com/v1/chat/completions",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "nano",
      stream: true,
      messages: [
        {
          role: "user",
          content: "Hello"
        }
      ]
    })
  }
)

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {

  const { done, value } =
    await reader.read()

  if (done) break

  console.log(
    decoder.decode(value)
  )
}
```
