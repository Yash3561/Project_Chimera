# streamlit_app.py (v3.0 - "THE FINAL CUT")

import streamlit as st
import os
import time
from agent import Agent

# --- Page Configuration ---
st.set_page_config(page_title="Chimera Workstation", page_icon="🤖", layout="wide")

# --- Agent Initialization ---
@st.cache_resource
def load_chimera_agent():
    print("Initializing Chimera Agent for Streamlit UI...")
    agent = Agent()
    print("Chimera Agent is ready.")
    return agent

chimera_agent = load_chimera_agent()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_history" not in st.session_state:
    st.session_state.log_history = "Agent log will appear here...\n"
if "terminal_history" not in st.session_state:
    st.session_state.terminal_history = "Script outputs will appear here...\n"

# --- Main UI Layout ---
st.title("🤖 Project Chimera Workstation")

col1, col2 = st.columns([1, 2])

# --- Left Column: Sandbox ---
with col1:
    st.header("Sandbox")
    
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, key="file_uploader_key")
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join("sandbox", file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success("File(s) uploaded!")
        time.sleep(1)
        st.rerun()

    st.write("---")
    
    file_list = sorted([f for f in os.listdir("sandbox")])
    if not file_list:
        st.info("Sandbox is empty.")
    else:
        selected_file = st.selectbox("Select a file to manage:", [""] + file_list, key="file_selector")
        if selected_file:
            file_path = os.path.join("sandbox", selected_file)
            if st.button(f"🗑️ Delete '{selected_file}'", use_container_width=True, type="primary"):
                os.remove(file_path)
                st.rerun()

            st.write(f"**Preview of `{selected_file}`:**")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text") # st.code is better for text/code
            except Exception:
                try:
                    st.image(file_path) # st.image for image files
                except:
                    st.warning("Cannot preview this file type.")

# --- Right Column: Agent Interaction ---
with col2:
    st.header("Agent Interaction")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main chat input
    if prompt := st.chat_input("Enter your objective..."):
        # Add user message to state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to show the user's message immediately

    # Display the agent's response if it's the last message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("🤔 Chimera is thinking..."):
                log_history = ""
                terminal_history = ""
                final_response = "Sorry, I reached my operational limit."

                # Collect all updates from the agent's run
                for update in chimera_agent.run_single_task(st.session_state.messages[-1]["content"]):
                    update_type, content = update["type"], update["content"]
                    if update_type == "log": log_history += content + "\n"
                    elif update_type == "terminal": terminal_history += content + "\n"
                    elif update_type == "chat": final_response = content
                
                # Display the final results
                st.markdown(final_response)
                st.session_state.log_history = log_history
                st.session_state.terminal_history = terminal_history

        # Add the final agent response to the message history
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun() # Rerun to lock in the agent's response

    # Accordions for logs and terminal
    with st.expander("Show Agent's Work (Detailed Log)"):
        # Use a code block to prevent markdown rendering issues
        st.code(st.session_state.log_history, language="text")
    
    if st.session_state.terminal_history.strip() != "Script outputs will appear here...":
         with st.expander("Show Terminal Output", expanded=True):
            st.code(st.session_state.terminal_history, language="text")

    if st.button("🔄 Reset Agent Memory & Chat", use_container_width=True):
        chimera_agent.reset_conversation()
        st.session_state.messages = []
        st.rerun()