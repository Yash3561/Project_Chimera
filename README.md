# Project Chimera - The Autonomous AI Workhorse

**This isn't just another AI agent. It's a resilient, multimodal, self-correcting reasoning engine built from the ground up to tackle complex, multi-step objectives in a persistent environment.**

This project was forged through rigorous, iterative debugging to create a truly robust agent architecture that overcomes common failure points seen in simpler prototypes. It is designed for stability, power, and genuine autonomy.

---

## Core Capabilities: See, Hear, Think, Act, Learn

Chimera is more than a language model in a loop. It's a complete system with a full suite of senses and tools.

*   👀 **See:** Utilizes **Florence-2-large** for state-of-the-art vision, allowing it to perform detailed OCR on full pages of text or generate rich descriptions of images.

*   🎧 **Hear:** Employs **distil-whisper** for fast, accurate audio transcription, enabling it to process information from audio files like meetings or recordings.

*   🧠 **Reason:** Powered by **Llama-3.1-8B-Instruct** through the high-performance **vLLM engine**, the agent uses a sophisticated **ReAct (Reason + Act)** loop to break down complex goals into a sequence of logical steps.

*   ⚡ **Act:** Wields a hardened, sandboxed toolset for interacting with its environment:
    *   **File System:** Full CRUDL (Create, Read, Update, Delete, List) operations, completely jailed to a secure `sandbox` directory.
    *   **Code Interpreter:** Writes and executes Python scripts in an isolated environment, capable of installing its own dependencies on the fly.
    *   **Shell Access:** Can run shell commands (`ls`, `cat`, etc.) directly in the sandbox for powerful system interactions.
    *   **Web Research Suite:** A multi-tool web stack featuring **Tavily** for AI-native search, a web page scraper for deep reading, and a binary file downloader.

*   📚 **Learn:** Features a persistent long-term memory powered by a **ChromaDB vector store**, allowing it to remember and recall facts across sessions.

---

## Architectural Highlights: Why This Agent is Different

This project showcases solutions to critical, real-world challenges in building autonomous agents.

### Resilient by Design
The agent is built to survive failure. It features a `max_consecutive_failures` counter that triggers a **human-in-the-loop (HITL)** fallback, forcing the agent to ask for help when it gets stuck, preventing infinite loops and wasted resources.

### Sandboxed & Secure
All operations that interact with the system are strictly sandboxed:
- The `FileSystemTool` is jailed to the `/sandbox` directory, preventing any possibility of path traversal attacks.
- The `CodeInterpreterTool` executes all user-generated code inside temporary, isolated directories that are destroyed after each run.

### Robust Data Handling Workflow
Through rigorous testing, a critical failure mode was identified: "context corruption" from handling large, messy data. Chimera solves this with a professional-grade workflow:
1.  Data-producing tools (`vision`, `audio`) **save their output directly to a file**.
2.  The agent **never touches the raw data directly.** It only ever handles clean, simple filenames.
3.  This prevents JSON parsing failures and keeps the agent's reasoning context clean and focused, dramatically increasing stability on complex tasks.

### High-Performance Inference
Instead of a basic Hugging Face pipeline, the core LLM runs on **vLLM**, a state-of-the-art serving engine that provides significantly higher throughput and lower latency, making the agent faster and more responsive.

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **Core Engine** | Python, vLLM |
| **LLM (Brain)** | `meta-llama/Llama-3.1-8B-Instruct` |
| **Vision Model** | `microsoft/Florence-2-large` |
| **Audio Model** | `distil-whisper/distil-medium.en` |
| **Long-Term Memory** | ChromaDB, Sentence-Transformers |
| **Web Search** | Tavily AI API |
| **Core Libraries** | `transformers`, `torch`, `requests`, `beautifulsoup4`|

---

## Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Yash3561/Project_Chimera.git
    cd Project_Chimera
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    -   Create a file named `.env` in the project root.
    -   Add your Tavily API key to it: `TAVILY_API_KEY="your-key-here"`

5.  **Run the agent:**
    ```bash
    python agent.py
    ```

You can now give the agent complex objectives directly in your terminal.
