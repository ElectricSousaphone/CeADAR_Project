# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 18:48:04 2025

This script retrieves information from the vector database and generates 
answers to the queries that are inputted into the pipeline.

@author: Shiv
"""


import os
from pinecone import Pinecone
from typing import List
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import time
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger
from dataclasses import dataclass

@dataclass
class Configurations:
    pinecone_index = "ceadar-documents1"
    pinecone_api_key = 'PINECONE_API_KEY'


class RAGPipeline:
    def __init__(
            self, 
            pinecone_api_key: str, 
            pinecone_index_name: str,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            inference_model: str ="Google/flan-t5-base"
        ):
        """
        Initialize RAG pipeline with Pinecone and Hugging Face configurations
        
        param pinecone_api_key: Pinecone API key
        param pinecone_index_name: Name of the Pinecone index
        param hf_model: Hugging Face model for embedding and QA
        """
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(pinecone_index_name)
        self.embedding_client = InferenceClient(model=embedding_model)
        self.generation_client = InferenceClient(model=inference_model)
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 20) -> List[str]:
        """
        Retrieve most relevant documents for a given query
        
        param query: Search query
        param top_k: Number of documents to retrieve
        return: List of most relevant document texts
        """
        # Generate query embedding
        query_embedding = False
        while query_embedding is False:
            try:
                query_embedding = self.embedding_client.feature_extraction(query)
            except HfHubHTTPError as e:
                logger.info(f"HTTP Error {e}")
                time.sleep(2)

        query_embedding = query_embedding.flatten().tolist()
        
        # Perform similarity search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k, 
            include_metadata=True
        )
        
        relevant_texts = [
            match['metadata']['text'] 
            for match in results['matches']
        ]
        
        return relevant_texts
    
    def create_qa_prompt(self, question, context) -> str:
        prompt = f"""
         
        Answer the question based on the given context in as much detail as possible.
        But also try to make the answer concise and accurate. Include all relevant 
        details from the context and Elaborate on each point. If there is not enough
        information in the context to answer the question reply 
        'There is not enough information in the documents'.
        
        
        Example:
    
        Question: How does the attention mechanism improve the performance of the Transformer model?
        Answer: The attention mechanism improves the performance of the Transformer model by allowing it to process all tokens in the input sequence simultaneously, rather than sequentially. 
        This parallel processing capability leads to faster training times and better handling of long-range dependencies.
        
        ---
        
        Question: What role does the attention mechanism play in the encoder-decoder architecture of the Transformer?
        Answer: In the encoder-decoder architecture of the Transformer, the attention mechanism helps the encoder to create context-aware representations of the input sequence.
        The decoder then uses these representations, along with its own attention mechanism, to generate the output sequence, ensuring that each output token is informed by the entire input sequence.
        
        Context: {context}
                
        Question: {question}

        Now Answer:"""
        
        return prompt
    
    def answer_question(self, query: str) -> str:
        """
        Answer a question using retrieved documents
        
        param query: Question to answer
        return: Generated answer
        """
        # Retrieve relevant documents
        context_docs = self.retrieve_relevant_documents(query)
        
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        context = " ".join(context_docs)
        
        answer = False
        while not answer:
            try:
                # Generate answer using a separate model
                answer = self.generation_client.text_generation(
                    prompt=self.create_qa_prompt(query, context),
                    max_new_tokens=250,
                    temperature=0.6,
                    repetition_penalty=1.5,
                    do_sample=True
                )
                logger.info(f"Query: {query}")
                logger.info(f"Answer: {answer}")
            except HfHubHTTPError as e:
                logger.info(f"HTTP Error {e}")
                time.sleep(2)
        
        return answer, context


def main():
    load_dotenv()
    
    cfg = Configurations()
    
    rag_pipeline = RAGPipeline(
        pinecone_api_key=os.getenv(cfg.pinecone_api_key),
        pinecone_index_name=cfg.pinecone_index,
    )
   

    query = "Explain how the attention head works"
    rag_pipeline.answer_question(query)   


if __name__ == "__main__":
    main()

