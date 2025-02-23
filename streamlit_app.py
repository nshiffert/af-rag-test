# streamlit_app.py

import streamlit as st
import streamlit.components.v1 as components
from backend import get_embedding, search_video_segments
import re
import openai

# Set OpenAI API key for ChatCompletion calls.
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

FULL_PROMPT_GLOBAL = ""
VIDEO_MAP = {}

# Utility functions for video URL/name formatting.
def transform_video_url(url: str) -> str:
    """Modify the Google Drive URL to use embed-friendly 'preview' mode."""
    if "drive.google.com" in url:
        return url.replace("view", "preview")
    return url

def transform_video_name(name: str) -> str:
    """Clean up the video name by replacing hyphens with spaces and removing file extensions."""
    name = name.replace("-", " ")
    if "." in name:
        name = name.split('.')[0]
    return name

# --- Chat Functions ---

def expand_query(query: str) -> str:
    """
    Use ChatGPT to expand the user's query into a detailed prompt.
    """
    return query
    """
    prompt = f"Expand the following query into a comprehensive, detailed prompt to help find training video segments: '{query}'"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that expands queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    expanded = response.choices[0].message["content"].strip()
    return expanded
    """

def get_chat_response(full_prompt: str, conversation: list) -> str:
    """
    Get a ChatGPT response using the conversation history plus the new prompt.
    """
    conversation.append({"role": "user", "content": full_prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.7,
    )
    assistant_response = response.choices[0].message["content"].strip()
    conversation.append({"role": "assistant", "content": assistant_response})
    return assistant_response

def extract_video_urls(text: str) -> list:
    """
    Look for markers in the assistant response that denote video references.
    Markers should be in the format [VIDEO: <url>].
    Returns a list of URLs.
    """
    pattern = r"\[VIDEO:\s*(.*?)\]"
    return re.findall(pattern, text)

# --- Session State Setup ---
# Always ensure the system prompt is what we want.
system_message = (
    "You are a helpful assistant who answers questions by integrating provided training video segments "
    "into your responses. Use the video segments as context to generate a detailed, practical answer. "
    "Try to include references to the videos in your response. If you make a video reference, "
    "include a marker in the format [VIDEO: <url>]."
)
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": system_message}]
else:
    # Update the system message in case it needs correction.
    st.session_state.conversation[0] = {"role": "system", "content": system_message}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (sender, message) tuples

st.title("Talk to your Training Videos")

# --- User Input Processing ---
with st.form(key="input_form"):
    user_input = st.text_input("Enter your question or message:","how do I create invoices?")
    submit_button = st.form_submit_button(label="Send")

if st.button("Clear Conversation"):
    st.session_state.conversation = [{"role": "system", "content": system_message}]
    st.session_state.chat_history = []
    st.stop()  # Refresh the page.

if submit_button and user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Processing your query..."):
        # Step 1: Expand the query.
        expanded_query = expand_query(user_input)
        
        # Step 2: Get embedding from OpenAI using the original query.
        embedding = get_embedding(user_input)
        
        # Step 3: Query Pinecone for matching video segments.
        raw_results = search_video_segments(embedding)
        # Filter results by score threshold (0.5).
        filtered_results = [r for r in raw_results if r.get("score", 0) >= 0.5]
        
        if filtered_results:
            video_context_lines = []
            for res in filtered_results:
                meta = res.get("metadata", {})
                text_content = meta.get("textContent", "")
                video_summary = meta.get("summary","")
                video_url = transform_video_url(meta.get("url", ""))
                video_name = transform_video_name(meta.get("videoName", "Unnamed Video"))
                score = res.get("score", "")
                line = f"Video: {video_name}\nScore: {score}\nContent: {text_content}\nLink: {video_url}"
                video_context_lines.append(line)
                VIDEO_MAP[video_url] = video_summary

            video_context = "\n\n".join(video_context_lines)
        else:
            video_context = "No relevant video segments were found."
        
        # Step 4: Build full prompt for ChatGPT.
        full_prompt = (
            f"Below are training video segments that might be relevant to the question:\n\n{video_context}\n\n"
            f"Based on the above segments, answer the following question in detail:\n{expanded_query}\n\n"
            "If no relevant segments are available, answer using your general training knowledge. "
            "Make sure your answer is comprehensive and include practical guidance. "
            "If you refer to a specific video, please include a marker in the format [VIDEO: <url>] where <url> is a valid"
            "url link to the video you are referring to."
        )
        FULL_PROMPT_GLOBAL = full_prompt
        
        # Step 5: Get ChatGPT's conversational response.
        assistant_message = get_chat_response(full_prompt, st.session_state.conversation)
    
    st.session_state.chat_history.append(("Assistant", assistant_message))

# --- Display Conversation History (after processing input) ---
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        # st.markdown(f"**PROMPT:** {FULL_PROMPT_GLOBAL}")
        st.markdown(f"**Assistant:** {message}")
        # Embed any videos referenced in the assistant's message.
        video_urls = extract_video_urls(message)
        for url in video_urls:
            components.iframe(url, width=300, height=300, scrolling=False)
            if(url in VIDEO_MAP):
                st.markdown(f"**Summary:** {VIDEO_MAP[url]}")
