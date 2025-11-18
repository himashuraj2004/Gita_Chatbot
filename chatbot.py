import os
import requests
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vector_db(persist_dir="gitadb"):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embed)
    return db


def generate_answer_with_openai(prompt, api_key, model="sonar"):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a Bhagavad Gita expert. Use ONLY the provided CONTEXT to answer the user's question."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    resp = r.json()
    return resp["choices"][0]["message"]["content"]


def build_prompt(context, query):
    return f"""
CONTEXT:
{context}

QUESTION: {query}

Answer concisely and cite verses where helpful.
"""


def chat_loop(db):
    # Load .env variables (if present) and then read API key
    load_dotenv()
    openai_key = os.environ.get("PPLX_API_KEY") or os.getenv("PPLX_API_KEY")
    print("Hare Krishna! (type 'exit' to quit).")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Hare Krishna!")
            break

        docs = db.similarity_search(q, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = build_prompt(context, q)

        if openai_key:
            try:
                answer = generate_answer_with_openai(prompt, openai_key)
            except Exception as exc:
                print("Error calling OpenAI:", exc)
                print("Falling back to showing retrieved context:")
                print(context[:2000])
                continue
        else:
            print("No API key set. Showing retrieved context instead:\n")
            print(context)
            continue

        print("\nAnswer:\n", answer)


def main():
    persist_dir = os.environ.get("GITA_DB_DIR", "gitadb")
    if not os.path.isdir(persist_dir):
        print(f"Persist directory '{persist_dir}' does not exist. Ensure your Chroma DB is at this path.")
        return
    db = load_vector_db(persist_dir)
    chat_loop(db)


if __name__ == "__main__":
    main()
