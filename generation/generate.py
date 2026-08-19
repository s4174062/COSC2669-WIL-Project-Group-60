"""
Takes retrieved chunks + a question, builds a strict grounded prompt,
and calls a local Ollama model.
"""

import ollama

MODEL_NAME = "llama3"

PROMPT_TEMPLATE = """You are a university policy assistant. Answer the question
using ONLY the context below. If the answer is not contained in the context,
say "I don't have enough information to answer that" rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    #test
    fake_context = [
        "Special Consideration applications must be submitted within two "
        "working days of the assessment date."
    ]
    answer = generate_answer(
        "How long do I have to apply for special consideration?", fake_context
    )
    print(answer)
