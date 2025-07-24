# tools/__init__.py (v3.1 - "MEMORY FIX")

from dotenv import load_dotenv
load_dotenv()

import subprocess
import os
import uuid
import logging
import shutil
import inspect
import chromadb
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

# Import all our individual tool classes
from .vision_tool import VisionTool
from .audio_tool import AudioTool
from .downloader_tool import DownloaderTool

# --- All Tool Classes Go Here ---
# ... (CodeInterpreterTool, FileSystemTool, WebSearchTool, WebReaderTool are all unchanged) ...
class CodeInterpreterTool:
    def __init__(self, workspace_dir="sandbox"):
        self.workspace_dir = os.path.abspath(workspace_dir)
    def execute_script(self, filename: str, args: list = None) -> str:
        script_path = os.path.join(self.workspace_dir, filename.lstrip('/\\'))
        if not os.path.exists(script_path): return f"Error: Script '{filename}' not found."
        try:
            python_executable = subprocess.check_output(["which", "python"]).strip().decode()
            command = [python_executable, script_path]
            if args: command.extend(args)
            process = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=self.workspace_dir, check=False)
            output = f"Script '{filename}' executed successfully.\nOutput:\n```\n{process.stdout}\n```"
            if process.returncode != 0: output += f"\nError:\n```\n{process.stderr}\n```"
            return output
        except Exception as e: return f"An unexpected error occurred: {e}"

class FileSystemTool:
    def __init__(self, workspace_dir="sandbox"):
        self.workspace_dir = os.path.abspath(workspace_dir)
    def _resolve_path(self, user_path: str) -> (str, str):
        abs_path = os.path.normpath(os.path.join(self.workspace_dir, user_path.lstrip('/\\')))
        if not abs_path.startswith(self.workspace_dir): return None, f"Error: Access denied. Path '{user_path}' is outside sandbox."
        return abs_path, None
    def write(self, filename: str, content: list[str] | str) -> str:
        abs_filepath, error = self._resolve_path(filename)
        if error: return error
        try:
            content_str = "\n".join(content) if isinstance(content, list) else str(content).replace('\\n', '\n')
            os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)
            with open(abs_filepath, 'w', encoding='utf-8') as f: f.write(content_str)
            return f"Successfully wrote to file '{filename}'."
        except Exception as e: return f"Error writing to file: {e}"
    def read(self, filename: str) -> str:
        abs_filepath, error = self._resolve_path(filename)
        if error: return error
        if not os.path.exists(abs_filepath): return f"Error: File '{filename}' not found."
        try:
            with open(abs_filepath, 'r', encoding='utf-8') as f: content = f.read()
            return f"Content of '{filename}':\n```\n{content}\n```"
        except Exception as e: return f"Error reading file: {e}"
    def peek(self, filename: str, lines: int = 10) -> str:
        abs_filepath, error = self._resolve_path(filename)
        if error: return error
        if not os.path.exists(abs_filepath): return f"Error: File '{filename}' not found."
        try:
            with open(abs_filepath, 'r', encoding='utf-8') as f:
                peek_content = "".join(f.readline() for _ in range(lines))
            return f"First {lines} lines of '{filename}':\n```\n{peek_content}\n```"
        except Exception as e: return f"Error peeking into file: {e}"
    def list(self, path: str = ".", recursive: bool = False) -> str:
        abs_path, error = self._resolve_path(path)
        if error: return error
        if not os.path.isdir(abs_path): return f"Error: Directory '{path}' not found."
        try:
            if not recursive: return f"Files in './{path}': {os.listdir(abs_path)}"
            else:
                output = f"Recursive file listing for './{path}':\n"
                for root, _, files in os.walk(abs_path):
                    relative_root = os.path.relpath(root, self.workspace_dir)
                    if relative_root == ".": relative_root = ""
                    level = relative_root.count(os.sep)
                    indent = "  " * level
                    output += f"{indent}./{os.path.basename(root)}/\n"
                    sub_indent = "  " * (level + 1)
                    for f in files: output += f"{sub_indent}- {f}\n"
                return output
        except Exception as e: return f"Error listing files: {e}"
    def delete(self, filename: str) -> str:
        abs_filepath, error = self._resolve_path(filename)
        if error: return error
        if not os.path.exists(abs_filepath): return f"Error: '{filename}' not found."
        try:
            if os.path.isdir(abs_filepath):
                shutil.rmtree(abs_filepath)
                return f"Successfully deleted directory '{filename}'."
            else:
                os.remove(abs_filepath)
                return f"Successfully deleted file '{filename}'."
        except Exception as e: return f"Error deleting '{filename}': {e}"

class WebSearchTool:
    def __init__(self):
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key: raise ValueError("TAVILY_API_KEY environment variable not set.")
        self.client = TavilyClient(api_key=api_key)
    def search(self, query: str, num_results: int = 5) -> str:
        try:
            response = self.client.search(query=query, search_depth="advanced", max_results=num_results)
            if not response or 'results' not in response: return "No results."
            results = [f"{i}. Title: {r.get('title', 'N/A')}\n   Snippet: {r.get('content', 'N/A')}\n   URL: {r.get('url', 'N/A')}" for i, r in enumerate(response['results'], 1)]
            return "Search Results:\n\n" + "\n\n".join(results)
        except Exception as e: return f"Error during search: {e}"

class WebReaderTool:
    def read(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)
            return text[:8000] if text else f"Error: No text found at URL: {url}"
        except Exception as e: return f"Error reading URL '{url}': {e}"

# <<< --- THE FIX IS IN THIS CLASS --- >>>
class MemoryTool:
    def __init__(self, db_path="agent_memory", collection_name="facts"):
        # 1. We still create the SentenceTransformer model here.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. We wrap our model in a class that ChromaDB understands.
        class MyEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, model):
                self.model = model
            def __call__(self, input_texts: list[str]) -> list[list[float]]:
                embeddings = self.model.encode(input_texts, convert_to_tensor=True)
                return embeddings.tolist()

        embedding_function = MyEmbeddingFunction(self.embedding_model)

        # 3. We connect to the database and get the collection, EXPLICITLY
        #    telling it to use our reliable embedding function.
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
    def remember(self, fact: str):
        try:
            # Now ChromaDB will use our custom function, not the broken ONNX one.
            self.collection.add(documents=[fact], ids=[str(uuid.uuid4())])
            return "Successfully remembered fact."
        except Exception as e: return f"Error remembering fact: {e}"

    def recall(self, query: str, num_results: int = 3):
        try:
            # This will also use our custom function.
            results = self.collection.query(query_texts=[query], n_results=num_results)
            recalled_docs = results['documents'][0]
            if not recalled_docs: return "I don't have any memories that match that query."
            return f"Recalled Memories:\n- " + "\n- ".join(recalled_docs)
        except Exception as e: return f"Error recalling memories: {e}"
# <<< --- END OF FIX --- >>>

# --- The Master Toolbox Dispatcher ---
class Toolbox:
    def __init__(self, workspace_dir="sandbox", memory_db_path="agent_memory"):
        logging.info("Initializing Toolbox...")
        self.tools = {
            "execute_python": CodeInterpreterTool(workspace_dir=workspace_dir),
            "filesystem": FileSystemTool(workspace_dir=workspace_dir),
            "web_search": WebSearchTool(),
            "web_reader": WebReaderTool(),
            "downloader": DownloaderTool(),
            "memory": MemoryTool(db_path=memory_db_path),
            "vision": VisionTool(),
            "audio": AudioTool(),
        }
        logging.info("Toolbox ready.")

    def use_tool(self, tool_name, args):
        if tool_name == "respond_to_user": return args.get("text", "...")
        if tool_name not in self.tools: return f"Error: Tool '{tool_name}' not found."

        logging.info(f"Using tool: {tool_name} with args: {args}")
        tool_instance = self.tools[tool_name]

        method_to_call = None
        if tool_name in ["filesystem", "memory", "execute_python", "vision"]:
            if 'operation' not in args: return f"Error: 'operation' argument missing."
            operation = args.pop('operation')
            if not hasattr(tool_instance, operation): return f"Error: Invalid operation '{operation}'."
            method_to_call = getattr(tool_instance, operation)
        elif tool_name == "audio": method_to_call = tool_instance.transcribe
        elif tool_name == "web_search": method_to_call = tool_instance.search
        elif tool_name == "web_reader": method_to_call = tool_instance.read
        elif tool_name == "downloader": method_to_call = tool_instance.download_file
        else: return f"Error: Dispatch logic not defined for '{tool_name}'."

        sig = inspect.signature(method_to_call)
        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
        
        try:
            return method_to_call(**valid_args)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"