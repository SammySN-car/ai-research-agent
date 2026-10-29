"""Interactive terminal interface for AI Research Agent."""

from langchain_core.messages import HumanMessage
from app.agent.graph import get_agent
from app.rag.ingestion import ingest_document


def main():
    """Initializes the AI Research Agent and displays the available terminal commands, including file upload and exit options. It enters a continuous loop that accepts user input, processing commands like /upload to ingest documents or sending general queries to the agent for retrieval and reasoning. The loop continues until the user explicitly types exit, quit, or q."""
    print("=" * 60)
    print("AI Research Agent - Terminal Chat Mode")
    print("Commands:")
    print("  /upload <path>   Upload & ingest a file (PDF, TXT, MD)")
    print("  exit / quit      Exit chat")
    print("=" * 60)

    print("\nLoading agent and embedding engine...")
    agent = get_agent()
    thread_id = "terminal-session"
    config = {"configurable": {"thread_id": thread_id}}
    print("Ready! Ask any question:\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            if user_input.startswith("/upload"):
                path = user_input.replace("/upload", "", 1).strip().strip('"').strip("'")
                print(f"Ingesting: {path}...")
                try:
                    doc = ingest_document(path)
                    print(f"[SUCCESS] Ingested {doc['filename']} ({doc['chunk_count']} chunks) into Supabase pgvector!")
                except Exception as e:
                    print(f"[ERROR] Ingestion failed: {e}")
                continue

            print("\n[Agent is retrieving & reasoning...]")
            result = agent.invoke({
                "messages": [HumanMessage(content=user_input)],
                "context": "",
                "tools_used": [],
            }, config=config)

            answer = result["messages"][-1].content
            tools_used = result.get("tools_used", [])

            print(f"\nAgent > {answer}")
            if tools_used:
                print(f"\n(Tools used: {', '.join(tools_used)})")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nSession closed. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()
