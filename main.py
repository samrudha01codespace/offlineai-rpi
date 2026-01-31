import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

MODEL_PATH = "/home/jarvis/LFM2.5-1.2B-Thinking-Q4_0.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_batch=64,
    n_gpu_layers=0,
    temperature=0.6,
    top_p=0.9,
    repeat_penalty=1.25,
    frequency_penalty=0.2,
    presence_penalty=0.2,
    n_threads=4,
    streaming=True,
    stop=["### User:", "### Instruction:"],
    verbose=False
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""### Instruction:
You are Jarvis, a helpful AI assistant. Answer clearly and concisely.

### User:
{question}

### Assistant:
"""
)

def ask(question: str):
    final_prompt = prompt.format(question=question)
    print("\n--- Jarvis ---")
    try:
        for chunk in llm.stream(final_prompt):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nError: {e}")
    print("\n----------------\n")

if __name__ == "__main__":
    print("\n🧠 Jarvis (Local GGUF) is running")
    print(f"📦 Model: {os.path.basename(MODEL_PATH)}")
    print("⌨️  Type 'exit' to quit")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("👋 Goodbye!")
            break
        if user_input:
            ask(user_input)
