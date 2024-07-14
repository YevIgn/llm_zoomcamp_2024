from openai import OpenAI


def hw2() -> None:
    prompt = "What's the formula for energy?"
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    print(response)


if __name__ == "__main__":
    hw2()
