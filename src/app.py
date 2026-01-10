import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import backend logic
from ingest import parse_construction_document
from agent_backend import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Construction Sidekick",
    layout="wide"
)

def main():
    st.title("Construction Sidekick")
    st.markdown("### AI Agent for BOQ/BOM Analysis")

    # Sidebar: File Upload
    with st.sidebar:
        st.header("Project Documents")
        uploaded_file = st.file_uploader("Upload BOM/BOQ (PDF)", type=["pdf"])
        
        # Reset button
        if st.button("Clear Chat Memory"):
            st.session_state.messages = []
            st.session_state.context = None
            st.rerun()

    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "context" not in st.session_state:
        st.session_state.context = None

    # File Processing
    if uploaded_file and st.session_state.context is None:
        with st.spinner("Agent is reading the blueprints... (This uses LlamaParse)"):
            # We need to save the uploaded file temporarily so LlamaParse can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Run the ingestion
                text_content = parse_construction_document(tmp_path)
                st.session_state.context = text_content
                st.success("Document processed! You can now ask questions.")
            except Exception as e:
                st.error(f"Error parsing document: {e}")
            finally:
                # Cleanup temp file
                os.remove(tmp_path)

    # Chat Interface
    
    # 1. Display existing chat history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    # 2. Chat Input
    # Only allow chatting if a file is uploaded and processed
    if st.session_state.context:
        user_input = st.chat_input("Ask about costs, materials, or quantities...")
        
        if user_input:
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add to history
            st.session_state.messages.append(HumanMessage(content=user_input))

            # Run Agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking & Calculating..."):
                    try:
                        # Prepare inputs
                        inputs = {
                            "messages": st.session_state.messages,
                            "context_data": st.session_state.context
                        }
                        
                        # Run Graph
                        result = agent_app.invoke(inputs)
                        response_content = result['messages'][-1].content
                        
                        st.write(response_content)
                        
                        # Update history with agent response
                        st.session_state.messages.append(AIMessage(content=response_content))
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a PDF to begin.")

if __name__ == "__main__":
    main()