import os
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# ------------------------------------------
# CONFIG
# ------------------------------------------
MODEL_PATH = "/Users/nic/Desktop/LFM2.5-1.2B-Thinking-Q4_0.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# ------------------------------------------
# LOAD LOCAL GGUF MODEL
# ------------------------------------------
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_batch=256,
    n_gpu_layers=0,
    temperature=0.4,          # LOWER
    top_p=0.9,
    repeat_penalty=1.25,      # 🔥 INCREASE
    frequency_penalty=0.3,    # 🔥 ADD
    presence_penalty=0.3,     # 🔥 ADD
    n_threads=os.cpu_count(),
    streaming=True,
    verbose=False
)


# ------------------------------------------
# PROMPT TEMPLATE
# ------------------------------------------
prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are Jarvis, a helpful AI assistant."""
)

# ------------------------------------------
# ASK FUNCTION
# ------------------------------------------
def ask(question: str):
    final_prompt = prompt.format(question=question)

    print("\n--- Jarvis ---")
    try:
        for chunk in llm.stream(final_prompt):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nError: {e}")

    print("\n----------------\n")


# ------------------------------------------
# MAIN LOOP
# ------------------------------------------
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
