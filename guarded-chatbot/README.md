# Guarded Chatbot (LangChain + LangGraph + NeMo Guardrails + OpenAI)

- **LangChain** – `ChatOpenAI` model and message utilities
- **LangGraph** – workflow with per-thread memory (`MemorySaver`)
- **NeMo Guardrails** – input/output self-check rails and Colang flows
- **OpenAI** – chat + embeddings (embeddings are used for Colang intent matching)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your OPENAI_API_KEY
python main.py
```

## Structure
```
main.py              CLI chat loop
app/config.py        settings (model, system prompt, history size)
app/graph.py         LangGraph workflow
app/guardrails.py    wraps the LangChain chain with NeMo Guardrails
config/config.yml    guardrails models + active rails
config/prompts.yml   self-check input/output prompts
config/rails.co      Colang flows (greeting, politics refusal)
```

## Customizing
- Change the chat model via `OPENAI_MODEL` in `.env` (self-check model is set in `config/config.yml`).
- Add policies in `config/prompts.yml`, and topics/flows in `config/rails.co`.
- Add more LangGraph nodes (tools, retrieval, routing) in `app/graph.py`.

Note: self-check rails make extra LLM calls per message (input + output), adding latency and cost.
