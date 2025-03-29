# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 17:48:04 2025

@author: Dhivaang
"""

import streamlit as st
import CeADAR_Interview as ci
import os
from dotenv import load_dotenv


def main():
    # Set up the Streamlit page
    st.set_page_config(page_title="Text Processor", page_icon="📝")
    
    # Create the Streamlit UI
    st.title("📝 Rag model CeADAR exercise")
    st.write("Please enter your question and click 'Answer question' to get a response. For example: Explain how the attention head works or What is the name of the DeepSeek R1 model?")
    
    # Text input area
    st.subheader("Question")
    user_text = st.text_area("Input Text:", height=100)
    
    load_dotenv()
    rag_pipeline = ci.RAGPipeline(
        pinecone_api_key=os.getenv('PINECONE_API_KEY'),
        pinecone_index_name="ceadar-documents"
    )
    
    # Process button
    if st.button("Answer question."):
        if not user_text:
            st.error("Please enter a question for the model to process!")
        else:
            with st.spinner("Processing..."):
                result = rag_pipeline.answer_question(user_text)
                
                # Display result
                st.subheader("Answer")
                st.text_area("Model Output:", result, height=100, disabled=True)
    
    # Footer
    st.markdown("---")
    st.markdown("A POC for a RAG model built by SMajithia")


if __name__=="__main__":
    main()