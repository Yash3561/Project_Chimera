# tools.py (Final Hardened Version)

import subprocess
import os
import uuid
import logging
import shutil
import inspect # We need this to check function arguments
import chromadb
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

# --- Hardened Tools ---

class CodeInterpreterTool:
    def __init__(self, workspace_dir="sandbox"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        if not os.path.exists(self.workspace_dir): os.makedirs(self.workspace_dir)
        logging.info(f"Code Interpreter using workspace: {self.workspace_dir}")
    def execute(self, code):
        try:
            run_id = str(uuid.uuid4())
            temp_exec_dir_base = os.path.join(os.path.dirname(self.workspace_dir), "temp_exec")
            os.makedirs(temp_exec_dir_base, exist_ok=True)
            exec_dir = os.path.join(temp_exec_dir_base, run_id)
            os.makedirs(exec_dir, exist_ok=True)
            script_path = os.path.join(exec_dir, "script.py")
            with open(script_path, 'w') as f: f.write(code)
            logging.info(f"Executing code in isolated directory: {exec_dir}")
            python_executable = subprocess.check_output(["which", "python"]).strip().decode()
            process = subprocess.run([python_executable, script_path], capture_output=True, text=True, timeout=30, cwd=self.workspace_dir, check=False)
            output = f"Code executed successfully.\nOutput:\n```\n{process.stdout}\n```" if process.returncode == 0 else f"Code execution failed with exit code {process.returncode}.\nError:\n```\n{process.stderr}\n```"
            shutil.rmtree(exec_dir)
            logging.info(f"Cleaned up temporary directory: {exec_dir}")
            return output
        except subprocess.TimeoutExpired: return "Error: Code execution timed out after 30 seconds."
        except Exception as e: return f"An unexpected error occurred during execution: {e}"

class FileSystemTool:
    def __init__(self, workspace_dir="sandbox"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        if not os.path.exists(self.workspace_dir): os.makedirs(self.workspace_dir)
    def _safe_path(self, path):
        full_path = os.path.abspath(os.path.join(self.workspace_dir, path))
        if not full_path.startswith(self.workspace_dir): raise ValueError("Access denied.")
        return full_path
    def write(self, filename, content):
        try:
            safe_filepath = self._safe_path(filename)
            os.makedirs(os.path.dirname(safe_filepath), exist_ok=True)
            with open(safe_filepath, 'w') as f: f.write(content)
            return f"Successfully wrote to file '{filename}'."
        except Exception as e: return f"Error writing to file: {e}"
    def read(self, filename):
        try:
            with open(self._safe_path(filename), 'r') as f: content = f.read()
            return f"Content of '{filename}':\n```\n{content}\n```"
        except Exception as e: return f"Error reading file: {e}"
    def list(self, path="."):
        try: return f"Files in '{path}': {os.listdir(self._safe_path(path))}"
        except Exception as e: return f"Error listing files: {e}"

class WebSearchTool:
    def search(self, query, num_results=5):
        logging.info(f"Performing web search for: '{query}'")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            if not results: return "No search results found."
            formatted_results = "Search Results:\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. Title: {result.get('title', 'N/A')}\n   Snippet: {result.get('body', 'N/A')}\n   URL: {result.get('href', 'N/A')}\n\n"
            return formatted_results
        except Exception as e: return f"Error during web search: {e}"

class MemoryTool:
    def __init__(self, db_path="agent_memory.db", collection_name="facts"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection(name=collection_name)
    def remember(self, fact):
        try:
            self.collection.add(embeddings=[self.embedding_model.encode(fact).tolist()], documents=[fact], ids=[str(uuid.uuid4())])
            return "Successfully remembered fact."
        except Exception as e: return f"Error remembering fact: {e}"
    def recall(self, query, num_results=3):
        try:
            results = self.collection.query(query_embeddings=[self.embedding_model.encode(query).tolist()], n_results=num_results)
            recalled_docs = results['documents'][0]
            if not recalled_docs: return "I don't have any memories that match that query."
            return f"Recalled Memories:\n- " + "\n- ".join(recalled_docs)
        except Exception as e: return f"Error recalling memories: {e}"

# --- The Hardened Toolbox Dispatcher ---
class Toolbox:
    def __init__(self, workspace_dir="sandbox", memory_db_path="agent_memory"):
        self.tools = {
            "execute_python": CodeInterpreterTool(workspace_dir=workspace_dir),
            "filesystem": FileSystemTool(workspace_dir=workspace_dir),
            "web_search": WebSearchTool(),
            "memory": MemoryTool(db_path=memory_db_path),
        }

    def use_tool(self, tool_name, args):
        if tool_name == "respond_to_user":
            return args.get("text", "I'm sorry, I had a thought but couldn't express it.")
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."

        logging.info(f"Using tool: {tool_name} with args: {args}")
        tool_instance = self.tools[tool_name]

        if hasattr(tool_instance, 'execute'): # For single-method tools like CodeInterpreter
            method_to_call = tool_instance.execute
        elif hasattr(tool_instance, 'search'): # For WebSearchTool
            method_to_call = tool_instance.search
        else: # For multi-method tools like filesystem and memory
            if 'operation' not in args:
                return f"Error: The 'operation' argument is missing for the '{tool_name}' tool."
            operation = args.pop('operation')
            if not hasattr(tool_instance, operation):
                return f"Error: Invalid operation '{operation}' for tool '{tool_name}'."
            method_to_call = getattr(tool_instance, operation)

        # --- The Anti-Hallucination Guard ---
        # Inspect the function's signature and filter args
        sig = inspect.signature(method_to_call)
        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        
        try:
            return method_to_call(**valid_args)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"