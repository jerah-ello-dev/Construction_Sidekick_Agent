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
    
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        # This instruction is critical for the "invisible lines" requirement
        parsing_instruction=(
            "This is a Construction Bill of Materials (BOM) or Bill of Quantities (BOQ). "
            "Extract all tabular data into valid Markdown tables. "
            "If gridlines are missing, infer columns based on text alignment. "
            "Ensure 'Description', 'Quantity', 'Unit', and 'Amount' columns are preserved. "
            "Capture section headers (like 'Ground Floor', 'Second Floor') clearly."
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