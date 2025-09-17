# app.py (v9.0 - "THE DEFINITIVE IDE")

import streamlit as st
import os
import time
from agent_ui import Agent # Make sure you're using the UI-refactored agent
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Chimera Workstation", page_icon="🤖", layout="wide")

# --- Agent Initialization ---
@st.cache_resource
def load_chimera_agent():
    print("Initializing Chimera Agent...")
    agent = Agent()
    print("Chimera Agent is ready.")
    return agent
chimera_agent = load_chimera_agent()

# --- Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "log_history" not in st.session_state: st.session_state.log_history = "Agent log will appear here...\n"
if "terminal_history" not in st.session_state: st.session_state.terminal_history = ""
if "selected_file" not in st.session_state: st.session_state.selected_file = None

# --- UI Helper Functions ---
def display_file_tree(path='sandbox', level=0):
    """A simple, clean, recursive file tree display."""
    indent = " " * 4 * level
    # Filter out junk files/folders
    items = sorted([item for item in os.listdir(path) if item not in ['.ipynb_checkpoints', '__pycache__']])
    
    for item in items:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            st.markdown(f"{indent}📁 **{item}/**")
            display_file_tree(full_path, level + 1)
        else:
            if st.button(f"{indent}📄 {item}", key=full_path, use_container_width=True):
                 st.session_state.selected_file = full_path
                 st.rerun()

# --- Main UI Layout ---
st.title("◎ Project Chimera Workstation")

# --- Sidebar: File Explorer & Management ---
with st.sidebar:
    st.header("Sandbox")
    with st.expander("File Management"):
        uploaded_file = st.file_uploader("Upload File")
        if uploaded_file:
            with open(os.path.join("sandbox", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded!"); time.sleep(1); st.rerun()

        new_file_name = st.text_input("New File Name")
        if st.button("Create New File"):
            if new_file_name: open(os.path.join("sandbox", new_file_name), 'a').close(); st.rerun()

    st.write("---")
    st.write("**File Explorer**")
    display_file_tree()
    
# --- Main Area: Tabbed Interface for Agent and Editor ---
agent_tab, editor_tab = st.tabs(["🤖 Agent Chat", "📝 File Editor"])

with agent_tab:
    st.header("Agent Interaction")
    chat_container = st.container(height=500, border=False)
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your objective..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with chat_container.chat_message("assistant"):
            with st.spinner("🤔 Chimera is working..."):
                final_response = None
                log_history = ""
                terminal_history = ""
                
                user_prompt = st.session_state.messages[-1]["content"]
                for update in chimera_agent.run_single_task(user_prompt):
                    update_type, content = update["type"], update["content"]
                    if update_type == "log": log_history += content + "\n"
                    elif update_type == "terminal": terminal_history += content + "\n"
                    elif update_type == "chat": final_response = content
                
                if final_response is None:
                    final_response = "Task complete. The agent did not provide a final verbal response. Check the logs for details."
                
                st.markdown(final_response)
                st.session_state.log_history = log_history
                st.session_state.terminal_history = terminal_history

        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun()

with editor_tab:
    st.header("File Editor")
    if st.session_state.selected_file:
        st.info(f"**Now editing:** `{st.session_state.selected_file}`")
        try:
            with open(st.session_state.selected_file, "r", encoding="utf-8") as f:
                content = f.read()
            edited_content = st.text_area("Content", content, height=500, key=f"editor_{st.session_state.selected_file}", label_visibility="collapsed")
            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                with open(st.session_state.selected_file, "w", encoding="utf-8") as f: f.write(edited_content)
                st.success("Saved!"); time.sleep(1)
        except Exception:
            st.image(st.session_state.selected_file)
    else:
        st.info("Select a file from the sidebar to view or edit it here.")

# --- Bottom Panel: Unified Logs & Terminal Output ---
st.write("---")
log_col, term_col = st.columns(2)

with log_col:
    with st.expander("Agent's Work (Detailed Log)"):
        st.code(st.session_state.log_history, language="text")

with term_col:
    with st.expander("Agent's Terminal Output", expanded=True):
        st.code(st.session_state.terminal_history, language="bash")

if st.sidebar.button("🔄 Reset Agent Memory & Chat"):
    chimera_agent.reset_conversation()
    st.session_state.messages = []
    st.rerun()