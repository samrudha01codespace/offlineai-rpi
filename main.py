import os
import sys
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

MODEL_PATH = "/home/jarvis/LFM2.5-1.2B-Thinking-Q4_0.gguf"

# Initialize the model with better parameters for Raspberry Pi
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,  # Increased from 2048 - minimum recommended for this model
    n_batch=32,  # Reduced for RPi memory constraints
    n_gpu_layers=0,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.15,  # Reduced from 1.25
    frequency_penalty=0.1,  # Reduced from 0.2
    presence_penalty=0.1,  # Reduced from 0.2
    n_threads=4,
    max_tokens=512,  # Limit response length
    streaming=True,
    stop=["### User:", "### Instruction:", "### Assistant:"],
    verbose=False,
    echo=False,  # Don't echo the prompt in output
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""### Instruction:
You are Jarvis, a helpful AI assistant. Answer clearly and concisely in one short paragraph.

### User:
{question}

### Assistant:
"""
)

# Conversation history management
conversation_history = []


def format_conversation(question, max_history=3):
    """Format conversation with limited history to manage context"""
    global conversation_history

    # Add new question to history
    conversation_history.append(f"User: {question}")

    # Keep only the last N exchanges
    if len(conversation_history) > max_history * 2:
        conversation_history = conversation_history[-max_history * 2:]

    # Format the prompt with history
    formatted_history = "\n".join(conversation_history[-max_history * 2:]) if len(
        conversation_history) > 1 else f"User: {question}"

    return f"""### Instruction:
You are Jarvis, a helpful AI assistant. Answer clearly and concisely based on the conversation history.

### Conversation History:
{formatted_history}

### User:
{question}

### Assistant:
"""


def ask(question: str):
    final_prompt = format_conversation(question)

    print("\n--- Jarvis ---")
    try:
        response_text = ""
        for chunk in llm.stream(final_prompt):
            print(chunk, end="", flush=True)
            response_text += chunk

        # Add response to history
        if response_text.strip():
            conversation_history.append(f"Assistant: {response_text.strip()}")

    except Exception as e:
        print(f"\nError: {e}")
        print("(Resetting conversation history due to error)")
        conversation_history.clear()  # Reset on error
    print("\n----------------\n")


if __name__ == "__main__":
    print("\n🧠 Jarvis (Local GGUF) is running")
    print(f"📦 Model: {os.path.basename(MODEL_PATH)}")
    print(f"🎯 Context window: {llm.n_ctx} tokens")
    print("⌨️  Type 'exit' to quit, 'clear' to reset conversation")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit", "bye"}:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "clear":
                conversation_history.clear()
                print("🗑️  Conversation history cleared")
                continue
            elif not user_input:
                continue

            ask(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")