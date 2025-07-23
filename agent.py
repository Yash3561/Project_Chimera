# agent.py (Final Version with ReAct Architecture)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import json
from tools import Toolbox # Make sure your tools.py is up-to-date

# --- Setup Logging ---
# This part is unchanged.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Mute a noisy library that often warns about padding tokens.
logging.getLogger("transformers.generation.utils").setLevel(logging.WARNING)


# --- The Brain (Unchanged) ---
# This class handles the low-level interaction with the LLM.
class LLM_Brain:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        logging.info(f"Initializing brain with model: {model_id}")
        self.model_id = model_id
        self.tokenizer = None
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Loads the tokenizer and model pipeline onto the GPU."""
        logging.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Define terminators to stop generation cleanly
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16, # Optimized for A100 GPUs
            device_map="auto",          # Automatically use the GPU
        )
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )
        logging.info("Brain is online and ready.")

    def get_decision(self, messages):
        """
        The core thinking process. Takes a conversation history and
        returns the agent's next thought and action.
        """
        logging.info("Brain is thinking...")
        # Re-enabling sampling to allow for more creative problem-solving
        outputs = self.pipe(
            messages,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            do_sample=True, 
            temperature=0.6, # A bit of creativity, but not too much
            top_p=0.9,
        )
        
        response = outputs[0]["generated_text"][-1]['content']
        logging.info(f"Brain's raw output:\n{response}")
        return response


class Agent:
    """
    The final, resilient agent using a ReAct architecture with
    self-correction and a human-in-the-loop fallback.
    """
    def __init__(self, max_loops=10, max_consecutive_failures=3):
        logging.info("Resilient ReAct Agent is booting up...")
        self.brain = LLM_Brain()
        self.toolbox = Toolbox(workspace_dir="sandbox", memory_db_path="agent_memory")
        self.max_loops = max_loops
        self.max_consecutive_failures = max_consecutive_failures
        self.system_prompt = self._get_react_system_prompt()

    def _get_react_system_prompt(self):
        """The final master prompt, now with explicit instructions on error handling."""
        tool_docs = json.dumps({
            "respond_to_user": "Responds to the user. Use this when you have the final answer or if you need to ask for clarification. args: {'text': 'Your final response.'}",
            "execute_python": "Executes Python code. args: {'code': 'Your Python code.'}",
            "filesystem": "Interacts with files. args: {'operation': '<write|read|list>', ...}",
            "web_search": "Searches the web. args: {'query': 'Your search query.'}",
            "memory": "Stores and retrieves info from long-term memory. args: {'operation': '<remember|recall>', ...}"
        }, indent=2)

        return f"""
You are Chimera, a highly capable reasoning agent. Your goal is to achieve the user's objective by breaking it down into a sequence of Thought-Action-Observation steps.

At each step, you must use the following JSON format:
{{
  "thought": "Your internal monologue. Analyze the situation, reflect on previous steps, and decide what to do next. **If the last observation was an error, you MUST analyze the error and form a new plan to fix it.**",
  "action": {{
    "tool": "tool_name",
    "args": {{"arg_name": "value"}}
  }}
}}

Your available tools:
{tool_docs}

Key Guidelines:
- Before using `web_search`, always `memory.recall` to see if you already know the answer.
- After finding key info, `memory.remember` it.
- **Error Handling:** If a tool returns an error, your next 'thought' must be about why the error occurred and how to correct your next action to solve it. Don't try the exact same action again.
- When you have the final answer, you MUST use `respond_to_user`.
- If you are stuck in a loop or repeatedly failing, use `respond_to_user` to ask the user for help.

Begin.
"""

    def run(self):
        """The main ReAct loop, now with a failure counter and human-in-the-loop fallback."""
        logging.info("Agent is running. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\nUser > ")
                if user_input.lower() == 'quit':
                    break

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"My objective is: {user_input}"}
                ]
                
                consecutive_failures = 0
                
                for loop_count in range(self.max_loops):
                    print(f"\n--- Loop {loop_count + 1}/{self.max_loops} (Failures: {consecutive_failures}) ---")
                    
                    # Check for human-in-the-loop trigger
                    if consecutive_failures >= self.max_consecutive_failures:
                        logging.warning("Max failures reached. Forcing human-in-the-loop.")
                        messages.append({"role": "user", "content": "Observation: You have failed multiple times in a row. You must ask the user for help. Use the 'respond_to_user' tool to describe the problem and ask for a suggestion."})
                        # Reset counter to give the agent a chance to ask for help
                        consecutive_failures = 0

                    llm_output = self.brain.get_decision(messages)
                    messages.append({"role": "assistant", "content": llm_output})
                    
                    try:
                        json_start = llm_output.find('{')
                        json_end = llm_output.rfind('}') + 1
                        if json_start == -1: raise ValueError("No JSON object found.")
                        clean_json_str = llm_output[json_start:json_end]
                        parsed_output = json.loads(clean_json_str)
                        thought = parsed_output.get("thought", "(No thought provided)")
                        action = parsed_output.get("action", {})
                        tool_name = action.get("tool")
                        tool_args = action.get("args", {})
                        print(f"Thought: {thought}")
                    except (json.JSONDecodeError, ValueError) as e:
                        error_message = f"Invalid JSON format. Error: {e}"
                        print(f"Chimera > {error_message}")
                        messages.append({"role": "user", "content": f"Observation: {error_message}"})
                        consecutive_failures += 1
                        continue

                    if not tool_name:
                        error_message = "You did not select a tool."
                        print(f"Chimera > {error_message}")
                        messages.append({"role": "user", "content": f"Observation: {error_message}"})
                        consecutive_failures += 1
                        continue

                    print(f"Action: {tool_name}({tool_args})")
                    tool_result = self.toolbox.use_tool(tool_name, tool_args)
                    print(f"Observation: {tool_result}")
                    
                    # Check for errors and update failure counter
                    if "Error:" in str(tool_result):
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0 # Reset on success

                    messages.append({"role": "user", "content": f"Observation: {tool_result}"})

                    if tool_name == "respond_to_user":
                        print("--- Objective Complete ---")
                        break
                else:
                    print("\n--- Agent reached max loops. ---")

            except Exception as e:
                logging.error(f"A critical error occurred: {e}", exc_info=True)

        logging.info("Agent is shutting down.")


if __name__ == "__main__":
    # Ensure you have applied the fix to tools.py before running this!
    agent = Agent()
    agent.run()