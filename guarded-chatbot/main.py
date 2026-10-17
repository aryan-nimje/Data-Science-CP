import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY in your .env file (see .env.example).")

from app.graph import build_graph  # noqa: E402


def main():
    graph = build_graph()
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Guarded chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        out = graph.invoke({"messages": [("user", user)]}, thread)
        print(f"Bot: {out['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
