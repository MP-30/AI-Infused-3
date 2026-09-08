import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv(find_dotenv(), override=True)

PROVIDER = "groq"
MODEL = "openai/gpt-oss-120b"
REQUIRED_KEY = "GROQ_API_KEY"
api_key = os.environ.get(REQUIRED_KEY)

if not api_key:
    raise SystemExit(
        f"{REQUIRED_KEY} not found in environment. "
        f"Add it to your .env file."
    )

print(
    f"Provider={PROVIDER} | "
    f"Model={MODEL} | "
    f"API key loaded={bool(api_key)}"
)

model = ChatGroq(
    model=MODEL,
    temperature=0,
    api_key=api_key,
)

# ---------------------------------------------------------------------------
# Tools
# Type hints:
#     Tell the model what arguments the tool expects.
# Docstrings:
#     Tell the model what the tool does and when it should be used.
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny, 28 degrees C."

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def get_population(city: str) -> str:
    """Get the approximate population of a major city."""
    data = {
        "new york": "8.5 million",
        "london": "9 million",
        "bangalore": "13 million",
        "tokyo": "14 million",
    }
    return data.get(
        city.lower(),
        f"Sorry, I don't have population data for {city}.",
    )

TOOLS = [get_weather, add, multiply, get_population,]

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=(
        "You are a helpful assistant. "
        "Use the available tools when needed. "
        "Always state the final result clearly and completely."
    ),
)

def ask(question: str, show_steps: bool = True) -> None:
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )
    print(f"\nQ: {question}")
    if show_steps:
        for message in response["messages"]:
            # AI requested a tool
            if (
                message.type == "ai"
                and getattr(message, "tool_calls", None)
            ):
                for tool_call in message.tool_calls:
                    print(
                        f"   [tool call] "
                        f"{tool_call['name']}({tool_call['args']})"
                    )
            # Tool returned a result
            elif message.type == "tool":
                print(
                    f"   [tool result] {message.content}"
                )
    print(f"A: {response['messages'][-1].content}")

if __name__ == "__main__":

    ask("What is the weather like in New York?")
    ask("What is infosys company")
    ask("What is 25 multiplied by 4?")
    ask("What is 137 plus 568?")
    ask("What is the population of Bangalore?")
    ask("What's the weather in Tokyo, and what is 12 times 12?")
    ask("Who wrote the play Romeo and Juliet?")