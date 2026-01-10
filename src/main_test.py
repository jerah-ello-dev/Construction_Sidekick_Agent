import os
import argparse
import sys

# Add the 'src' directory to Python path so we can import modules easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest import parse_construction_document
from agent_backend import app
from langchain_core.messages import HumanMessage

def main():
    # Setup Argument Parser to accept --file
    parser = argparse.ArgumentParser(description="Sprout Construction Sidekick")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF BOM file")
    args = parser.parse_args()

    print("--- Sprout Construction Sidekick (Terminal Mode) ---")
    
    file_path = args.file
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at: {file_path}")
        return

    print(f"Step 1: Ingesting Document: {file_path}")
    print("        (This may take 10-20 seconds via LlamaParse...)")
    
    try:
        context_text = parse_construction_document(file_path)
    except Exception as e:
        print(f"❌ Parsing Error: {e}")
        return
    
    print("\n--- Document Loaded. Agent is ready. Type 'exit' to quit. ---\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            chat_history.append(HumanMessage(content=user_input))
            
            inputs = {
                "messages": chat_history,
                "context_data": context_text
            }
            
            print("\n(Thinking...)\n")
            
            # Run the agent
            result = app.invoke(inputs)
            last_message = result['messages'][-1]
            
            print(f"Agent: {last_message.content}\n")
            
            # Update history
            chat_history = result['messages']
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during chat: {e}")

if __name__ == "__main__":
    main()