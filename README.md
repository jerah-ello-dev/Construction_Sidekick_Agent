# Construction BOQ Assistant (Agentic RAG)

An AI-powered agent designed to assist Civil Engineers and Quantity Surveyors by automating the analysis of Bill of Quantities (BOQ) and Bill of Materials (BOM) documents.

Unlike standard chatbots, this agent uses **Agentic RAG** (Retrieval Augmented Generation) combined with **Python REPL** tools to ensure mathematical accuracy. It specifically solves the "messy PDF" problem common in construction by using Vision-based parsing to read tabular data with invisible gridlines.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-green)
![LangGraph](https://img.shields.io/badge/LangGraph-🕸️-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-🦙-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

## Key Features

* **Advanced PDF Parsing:** Uses **LlamaParse** (Vision AI) to reconstruct messy tables, scanned documents, and BOQs where gridlines are invisible or broken.
* **Zero-Hallucination Math:** The agent does *not* calculate numbers in its "head." It writes and executes **Python code** locally to perform accurate cost estimations and material summations.
* **Agentic Reasoning:** Built on **LangGraph**, the agent creates a stateful workflow that can plan, reason, and switch between "reading" (Retrieval) and "calculating" (Tools).
* **Context-Aware Chat:** Remembers conversation history (e.g., "What about the second floor?") for natural interactions.

## Architecture

The system follows a **ReAct (Reason + Act)** architecture:

1.  **Ingestion:** User uploads a PDF -> LlamaParse converts it to structured Markdown.
2.  **Router:** The LLM decides if the user's query requires *Retrieval* (looking up facts) or *Calculation* (math).
3.  **Tool Execution:** If calculation is needed, the agent generates Python code (Pandas) and executes it in a sandboxed environment.
4.  **Response:** The final answer is synthesized from the tool output and natural language.

## Tech Stack

* **LLM Engine:** Llama 3.3 70B (via Groq API) for high-speed reasoning.
* **Orchestration:** LangGraph & LangChain.
* **Document Parsing:** LlamaParse (LlamaIndex ecosystem).
* **UI:** Streamlit.

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* API Key for **Groq** (LLM inference)
* API Key for **LlamaCloud** (PDF Parsing)

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/construction-boq-assistant.git](https://github.com/yourusername/construction-boq-assistant.git)
cd construction-boq-assistant

```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# If requirements.txt is missing, run:
pip install langchain langchain-groq langchain-community langchain-experimental langgraph llama-parse python-dotenv pandas tabulate streamlit watchdog

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=gsk_...
LLAMA_CLOUD_API_KEY=llx-...

```

## Usage

Run the web interface using Streamlit:

```bash
streamlit run src/app.py

```

1. Open the URL displayed in your terminal (usually `http://localhost:8501`).
2. Upload a **BOQ/BOM PDF** in the sidebar.
3. Wait for the parsing notification ("Document processed!").
4. Ask questions like:
* *"What is the total cost of cement listed in the file?"*
* *"List all materials needed for the Ground Floor."*



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

```

