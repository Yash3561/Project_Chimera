# Project Chimera - An Autonomous AI Agent

Project Chimera is a powerful, open-source AI agent built from the ground up. It uses a ReAct (Reason + Act) architecture to dynamically solve complex problems.

## Core Capabilities

- **Dynamic Reasoning:** Utilizes a `Thought-Action-Observation` loop to adapt its strategy in real-time.
- **Multi-Tool Proficiency:** Equipped with a versatile toolbox for:
  - **Code Execution:** Securely runs Python code in an isolated environment.
  - **Web Search:** Accesses real-time information from the internet.
  - **File System I/O:** Reads from and writes to a designated workspace.
  - **Long-Term Memory:** Learns and recalls information across sessions using a persistent vector database (ChromaDB).
- **Resilience:** Features a self-correction loop to analyze and recover from errors.
- **Human-in-the-Loop:** Knows when it's stuck and can ask the user for guidance.

## Architecture

This agent is built using a modular, Python-based framework.
- **`agent.py`:** Contains the main `Agent` class and the ReAct reasoning loop.
- **`tools.py`:** Defines the agent's capabilities (the `Toolbox`).
- **LLM Core:** Powered by Hugging Face's `transformers` library, designed to run models like Llama 3.

## Setup & Installation

1.  **Create a Conda Environment:**
    ```bash
    conda create -n chimera python=3.10 -y
    conda activate chimera
    ```

2.  **Install Dependencies:** Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.

3.  **Hugging Face Login:**
    ```bash
    huggingface-cli login
    ```

4.  **Run the Agent:**
    ```bash
    python agent.py
    ```

## How to Use

Simply provide a high-level objective at the prompt, and the agent will formulate and execute a plan to achieve it.