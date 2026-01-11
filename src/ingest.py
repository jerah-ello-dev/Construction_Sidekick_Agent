import os
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Apply nest_asyncio to allow async execution in Jupyter/Scripts if needed
nest_asyncio.apply()

def parse_construction_document(file_path):
    """
    Uses LlamaParse to extract text and tables from construction documents.
    It specifically instructs the model to look for BOM/BOQ structures.
    """
    print(f"--- Parsing document: {file_path} ---")

    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        raise ValueError(" Error: LLAMA_CLOUD_API_KEY is missing from .env file")
    
    # Initialize Parser
    parser = LlamaParse(
        api_key=llama_key,
        result_type="markdown",
        verbose=True,
        language="en",
        # This is the universal instruction
        parsing_instruction=(
            "Extract all text from this construction document.\n"
            "1. PRESERVE STRUCTURE: Keep all Article numbers (e.g., 'Article 8'), Section numbers (e.g., '8.1'), and Page numbers.\n"
            "2. DO NOT SUMMARIZE: Keep the legal text exactly as written.\n"
            "3. TABLES: Convert any BOM/BOQ tables into Markdown format with columns for Item, Description, Qty, Unit, and Amount.\n"
        )
    )

    documents = parser.load_data(file_path)
    
    # Combine all pages into one text block for the context
    full_text = "\n\n".join([doc.text for doc in documents])
    
    print("--- Parsing Complete ---")
    return full_text

if __name__ == "__main__":
    # Test block (Create a dummy file first if you want to test this alone)
    pass