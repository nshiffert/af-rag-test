# streamlit_app.py

import streamlit as st
import streamlit.components.v1 as components
from backend import get_embedding, search_video_segments
import datetime

def transform_video_url(url: str) -> str:
    """
    Modify the Google Drive URL to use the embed-friendly 'preview' mode.
    """
    if "drive.google.com" in url:
        return url.replace("view", "preview")
    return url

def transform_video_name(name: str) -> str:
    """
    Convert a hyphenated video name into a cleaner format by replacing hyphens with spaces
    and removing file extensions.
    """
    name = name.replace("-", " ")
    if "." in name:
        name = name.split('.')[0]
    return name

st.title("Video Training Assistant")

# Input widget for the user's question
question = st.text_input("Enter your question:")

if st.button("Search"):
    if question:
        with st.spinner("Processing your query..."):
            # Get embedding from OpenAI
            embedding = get_embedding(question)
            # Query Pinecone for matching video segments
            results = search_video_segments(embedding)
        
        if results:
            st.write("### Results:")
            for res in results:
                meta = res.get("metadata", {})
                content_text = meta.get("textContent", "No content text provided.")
                video_url = meta.get("url", "")
                video_created = meta.get("videoCreated", "Unknown date")
                video_duration = meta.get("videoDuration", "Unknown duration")
                video_name = meta.get("videoName", "Unnamed Video")

                # Clean up the video URL and name
                video_url = transform_video_url(video_url)
                video_name = transform_video_name(video_name)

                # Display result details
                st.markdown(f"#### {video_name}")
                st.write(f"**Created:** {video_created} | **Duration:** {video_duration}")
                st.write(content_text)

                # Display embedded video using Streamlit's iframe component
                components.iframe(video_url, width=300, height=300, scrolling=False)
        else:
            st.write("No results found for your query.")
    else:
        st.warning("Please enter a question before searching.")
