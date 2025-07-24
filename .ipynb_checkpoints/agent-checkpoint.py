# agent.py (v11.0 - "CHIMERA FINALIZED")

from vllm import LLM, SamplingParams 
import logging
import json
import re
from tools import Toolbox

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("vllm").setLevel(logging.WARNING)

class LLM_Brain:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        logging.info(f"Initializing brain with vLLM engine: {model_id}")
        self.llm = LLM(model=model_id, dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.90)
        logging.info("Brain is online and ready.")

    def get_decision(self, messages: list) -> str:
        logging.info("Brain is thinking...")
        prompt = self.llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True,
        )
        stop_token = self.llm.llm_engine.tokenizer.tokenizer.eos_token
        params = SamplingParams(
            temperature=0.4, top_p=0.9, max_tokens=1024, stop=[stop_token, "\n}\n"]
        )
        outputs = self.llm.generate([prompt], params)
        response = outputs[0].outputs[0].text.strip()
        logging.info(f"Brain's raw output:\n{response}")
        return response

class Agent:
    def __init__(self, max_loops=15, max_consecutive_failures=3):
        logging.info("Resilient ReAct Agent is booting up...")
        self.brain = LLM_Brain()
        self.toolbox = Toolbox()
        self.max_loops = max_loops
        self.max_consecutive_failures = max_consecutive_failures
        self.system_prompt = self._get_react_system_prompt()

    def _get_react_system_prompt(self):
        """
        The final, v11.0 "bulletproof" system prompt that teaches the agent a
        robust data handling workflow to prevent context corruption.
        """
        tool_docs_dict = {
            "respond_to_user": "Responds to the user when the task is complete. args: {'text': 'Your final response.'}",
            "execute_python": "Executes a Python script from the sandbox. args: {'operation': 'execute_script', 'filename': 'your_script.py', 'args': ['arg1']}",
            "filesystem": "Interacts with files. For writing code, provide 'content' as a list of strings. args: {'operation': '<write|read|list|delete|peek>', 'filename': 'file.py', 'content': ['line1', 'line2']}",
            "web_search": "Searches the web for URLs. args: {'query': 'Your search query.'}",
            "web_reader": "Reads the text content of a URL. args: {'operation': 'read', 'url': 'https://...'}",
            "downloader": "Downloads a file from a URL. args: {'url': 'https://.../data.zip', 'filename': 'data.zip'}",
            "memory": "Remembers or recalls facts. args: {'operation': '<remember|recall>', 'fact': '...', 'query': '...'}",
            "vision": "Analyzes an image file. Use '<OCR>' for full-page text extraction. Use '<MORE_DETAILED_CAPTION>' for image descriptions. args: {'image_path': 'path/to/image.png', 'task_prompt': '<OCR>'}",
            "audio": "Transcribes an audio file. args: {'audio_path': 'path/to/audio.mp3'}"
        }
        tool_docs = json.dumps(tool_docs_dict, indent=2)

        return f"""You are Chimera, an AI agent. You have one job: to output a single, valid JSON object that follows a strict schema. Adhering to this JSON schema is the most important rule.

**YOUR JSON RESPONSE MUST USE THIS EXACT STRUCTURE AND NOTHING ELSE:**
```json
{{
  "thought": "Your concise plan for the single next action. THIS IS MANDATORY.",
  "action": {{
    "tool": "The single tool you will use.",
  "args": {{
      "argument_name": "argument_value"
    }}
  }}
}}
```
**Do not add any text before or after this JSON. Do not change the structure.**

Your available tools:
{tool_docs}


**CRITICAL RULE FOR DATA HANDLING:**
1.  When a tool (`vision`, `web_reader`, `audio`) returns a large block of text, you are *FORBIDDEN* from putting that text directly into your next 'thought' or 'action'.
Your *ONLY* allowed next step is to save the text to a file.
2.  To analyze or use that content later, read it back from the file using `filesystem` with operation `read`.
3.  *NEVER* put large, multi-line blocks of text into your `respond_to_user` tool call. Instead, summarize the content of the file you saved, or tell the user that the full text is available in the file.


**Example Workflows:**
**Web Research Workflow:**
1.  Use `web_search` to find relevant URLs for a topic.
2.  Use `web_reader` with a specific URL from the search results to read its content.
3.  Use `downloader` to save a file.
4.  Use `memory` with operation `remember` to store key facts you learned.
5.  Use `filesystem` with operation `peek` to verify downloads.
**Coding:**
1.  Use `filesystem` with operation `write` with `content` as a LIST OF STRINGS to create a `.py` script in the sandbox.
2.  Use `execute_python` with operation `execute_script` to run the file.

Begin.
"""

    def _launder_json(self, json_string: str) -> str:
        """
        Brutally cleans a string to make it JSON-parsable.
        Handles unescaped quotes and newlines inside string values.
        This is a last resort to fix a malformed JSON string from the LLM.
        """
        # Add backslashes to escape all unescaped double quotes
        # but be careful not to double-escape already escaped quotes (e.g., \\")
        json_string = re.sub(r'(?<!\\)"', r'\\"', json_string)

        # Now that all quotes are escaped, we can remove the outer quotes
        # that the LLM might have added around the whole block
        if json_string.startswith(r'\"') and json_string.endswith(r'\"'):
            json_string = json_string[2:-2]
            
        # Replace newline characters with the escaped version
        json_string = json_string.replace('\n', '\\n')

        return json_string

    def run(self):
        logging.info("Agent is running. Type 'quit' to exit or '/reset' to clear history.")
        messages = [{"role": "system", "content": self.system_prompt}]
        
        while True:
            try:
                user_input = input("\nUser > ")
                if user_input.lower() == 'quit': break
                if user_input.lower() == '/reset':
                    print("--- Resetting conversation history. ---")
                    messages = [{"role": "system", "content": self.system_prompt}]
                    continue
    
                messages.append({"role": "user", "content": f"My objective is: {user_input}"})
                consecutive_failures = 0
                
                for loop_count in range(self.max_loops):
                    print(f"\n--- Loop {loop_count + 1}/{self.max_loops} (Failures: {consecutive_failures}) ---")
                    
                    if consecutive_failures >= self.max_consecutive_failures:
                        logging.warning("Max failures reached. Forcing human-in-the-loop.")
                        human_intervention_prompt = "Observation: You have failed multiple times. You MUST ask the user for help. Use 'respond_to_user' to describe the problem and ask for a suggestion."
                        messages.append({"role": "user", "content": human_intervention_prompt})
                        consecutive_failures = 0
    
                    llm_output = self.brain.get_decision(messages)
                    
                    try:
                        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                        if not json_match:
                            raise ValueError("No JSON object found in LLM output.")
                        
                        parsed_output = json.loads(json_match.group(0))
                        
                        thought = parsed_output.get("thought", "(No thought provided)")
                        action = parsed_output.get("action", {})
                        tool_name = action.get("tool")
                        tool_args = action.get("args", {})
                        
                        print(f"Thought: {thought}")
    
                    except (json.JSONDecodeError, ValueError, AttributeError) as e:
                        error_message = f"Your response was not valid JSON. You MUST follow the specified format. Error: {e}\nYour output was:\n{llm_output}"
                        print(f"Chimera > PARSING ERROR: {error_message}")
                        messages.append({"role": "user", "content": f"Observation: {error_message}"})
                        consecutive_failures += 1
                        continue
    
                    if not tool_name:
                        error_message = "Your JSON is missing the 'tool' key."
                        print(f"Chimera > LOGIC ERROR: {error_message}")
                        messages.append({"role": "user", "content": f"Observation: {error_message}"})
                        consecutive_failures += 1
                        continue
    
                    messages.append({"role": "assistant", "content": json.dumps(parsed_output, indent=2)})
    
                    print(f"Action: {tool_name}({tool_args})")
                    tool_result = self.toolbox.use_tool(tool_name, tool_args)
                    
                    observation = ""
                    if "Error:" in str(tool_result):
                        consecutive_failures += 1
                        observation = f"The tool returned an error. Analyze this and try a different approach. Observation: {tool_result}"
                    else:
                        consecutive_failures = 0
                        observation = str(tool_result)
    
                    print(f"Observation: {observation}")
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
    
                    if tool_name == "respond_to_user":
                        print("--- Objective Complete ---")
                        break
                else:
                    print("\n--- Agent reached max loops for this objective. ---")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Shutting down.")
                break
            except Exception as e:
                logging.error(f"A critical error occurred: {e}", exc_info=True)
                break
        logging.info("Agent is shutting down.")


if __name__ == "__main__":
    agent = Agent()
    agent.run()