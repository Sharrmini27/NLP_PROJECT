import streamlit as st
from transformers import pipeline
from newspaper import Article
import time

# Page setup for the assignment
st.set_page_config(page_title="NLP News Summarizer", page_icon="📰")

# Load the HuggingFace BART model (Abstractive Summarization)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

st.title("📰 NLP Project: News Summarizer")
st.markdown("This app uses a **Transformer (BART)** model to generate summaries.")

# Input URL
url = st.text_input("Enter News Article URL:")

if st.button("Summarize"):
    if url:
        try:
            with st.spinner('Processing...'):
                start_time = time.time()
                
                # Scrape article
                article = Article(url)
                article.download()
                article.parse()
                
                # Execute Summarization
                summary_output = summarizer(article.text, max_length=150, min_length=50, do_sample=False)
                summary_text = summary_output[0]['summary_text']
                
                end_time = time.time()
                
                # Display Results
                st.subheader("Summary")
                st.success(summary_text)
                
                # Performance Measurements (Required by Rubric)
                orig_len = len(article.text.split())
                summ_len = len(summary_text.split())
                duration = round(end_time - start_time, 2)
                
                st.write("---")
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Words", orig_len)
                col2.metric("Summary Words", summ_len)
                col3.metric("Processing Time", f"{duration}s")
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a URL.")
