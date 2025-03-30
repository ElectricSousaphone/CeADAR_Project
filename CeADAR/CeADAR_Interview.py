# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 18:48:04 2025

@author: Shiv
"""


import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import time
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger
from dataclasses import dataclass

@dataclass
class Configurations:
    filepaths = [
        'EU AI Act Doc.docx', 
        'Deepseek-r1.pdf', 
        'Attention_is_all_you_need.pdf'
    ]
    pinecone_index = "ceadar-documents"
    pinecone_api_key = 'PINECONE_API_KEY'
    load_data = False


class RAGPipeline:
    def __init__(
            self, 
            pinecone_api_key: str, 
            pinecone_index_name: str,
            hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        """
        Initialize RAG pipeline with Pinecone and Hugging Face configurations
        
        param pinecone_api_key: Pinecone API key
        param pinecone_index_name: Name of the Pinecone index
        param hf_model: Hugging Face model for embedding and QA
        """
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        #pc.delete_index(name=pinecone_index_name)
        #pc.create_index(name=pinecone_index_name, dimension=384, 
        #    spec=ServerlessSpec(cloud='aws', region='us-east-1')
        #)
        self.index = pc.Index(pinecone_index_name)
        self.embedding_client = InferenceClient(model=hf_model)
        self.generation_client = InferenceClient(model="Google/flan-t5-base")
        
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        
        param documents: List of text documents
        return embeddings: List of embeddings
        """
        
        embeddings = []
        for text in documents:
            
            emb = False
            while emb is False:
                try:
                    emb = self.embedding_client.feature_extraction(text)
                except HfHubHTTPError:
                    time.sleep(2)
            
            emb = emb.flatten().tolist()
            
            embeddings.append(emb)
        
        return embeddings
    
    def upsert_documents(self, documents: List[str]) -> None:
        """
        Embed and upsert documents into Pinecone with small batch sizes
        
        param documents: List of text documents to index
        """
        # Process in small document batches to avoid large embedding batches
        embeddings = self.embed_documents(documents)
        
        # Create vectors with unique IDs
        vectors = [
            (str(idx), 
             embeddings[idx],
             {"text": doc}) 
            for idx, doc in enumerate(documents)
        ]
        
        vector_count = len(vectors)
        
        if vector_count>1000:
            for i in range(0, vector_count, 250):
                self.index.upsert(vectors[i:i+250])
        else:
            # Upsert to Pinecone
            self.index.upsert(vectors)
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 20) -> List[str]:
        """
        Retrieve most relevant documents for a given query
        
        param query: Search query
        param top_k: Number of documents to retrieve
        return: List of most relevant document texts
        """
        # Generate query embedding
        query_embedding = self.embedding_client.feature_extraction(query)
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
                    temperature=0.8,
                    repetition_penalty=1.1,
                    do_sample=True
                )
                print(f"Query: {query}")
                print(f"Answer: {answer}")
                #print(f"Context: {context}")
            except HfHubHTTPError as e:
                logger.info(f"HTTP Error {e}")
                time.sleep(2)
        
        return answer


class LoadData:
    
    def __init__(self):
        pass
    
    def read_pdf(self, file_path: str) -> str:
        """
        This function reads in a pdf and converts it into string
        
        param filepath: The filepath of the pdf file
        return text: The document in string format
        
        """
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
                text = text.replace("\n", " ")
        return text
    
    def read_docx(self, file_path: str) -> str:
        """
        This function reads in a docx and converts it into string
        
        param filepath: The filepath of the docx file
        return text: The document in string format
        
        """
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + " "
            text = text.replace("\n", " ")
        return text
    
    def read_txt(file_path: str) -> str:
        """
        This function reads in a text file and converts it into a string
        
        param filepath: The filepath of the txt file
        return text: The document in string format
        
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    
    def load_and_preprocess_data(self, file_path: str) -> List:
        """
        This function reads in a files and splits it up into a list
        
        param filepath: The filepath of the txt file
        return texts: The document split up into strings within a list
        
        """
        
        if file_path.endswith('.pdf'):
            text_document = self.read_pdf(file_path)
        elif file_path.endswith('.docx'):
            text_document = self.read_docx(file_path)
        elif file_path.endswith('.txt'):
            text_document = self.read_txt(file_path)
    
        logger.info(f"Splitting text from {file_path} \n")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=[". ", "! ", "? ", "\n", " ", ""],
            chunk_overlap=15
        )
        texts = text_splitter.split_text(text_document) 
    
        return texts 
    
    def process_documents(self, file_paths: str) -> List:
        """
        This function reads the different files, preprocesses them and add them
        to a list
        
        param filepath: The filepath of the txt file
        return document: The document split up into strings within a list
        
        """        
        documents = []
        load_data = LoadData()
        for file_path in file_paths:
            for i in load_data.load_and_preprocess_data(file_path):
                documents.append(i.replace("\n", " "))
            logger.info(f"Data from {file_path} has been processed \n")
            
        return documents
    

def main():
    load_dotenv()
    
    cfg = Configurations()
    
    rag_pipeline = RAGPipeline(
        pinecone_api_key=os.getenv(cfg.pinecone_api_key),
        pinecone_index_name=cfg.pinecone_index,
    )
   
    if cfg.load_data:
        load_data = LoadData()
    
        documents = load_data.process_documents(cfg.filepaths)
        
        rag_pipeline.upsert_documents(documents)

    query = "What is the EU AI Act about what does it contain?"
    rag_pipeline.answer_question(query)   


if __name__ == "__main__":
    main()

