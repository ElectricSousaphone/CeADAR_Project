# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 18:48:04 2025

This script reads in data from the documents and loads them into the vector db

@author: Shiv
"""


import os
from pinecone import Pinecone, ServerlessSpec
from typing import List
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
        'Input_documents/EU AI Act Doc.docx', 
        'Input_documents/Deepseek-r1.pdf', 
        'Input_documents/Attention_is_all_you_need.pdf'
    ]
    pinecone_index = "ceadar-documents1"
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    delete_and_create_vect_db = True


class PopulateVectorDbPipeline:
    """ This pipeline populates the vector database"""
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
        self.pinecone_index_name = pinecone_index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedding_client = InferenceClient(model=hf_model)
        self.generation_client = InferenceClient(model="Google/flan-t5-base")
    
    def delete_and_create_db(self, delete_and_create_vect_db):
        
        index_names = [indexes["name"] for indexes in self.pc.list_indexes()]
        if self.pinecone_index_name in index_names:
            logger.info("Deleting the index from the database")
            self.pc.delete_index(name=self.pinecone_index_name)
        
        logger.info("Creating index within the database")
        self.pc.create_index(name=self.pinecone_index_name, dimension=384, 
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        
        param documents: List of text documents
        return embeddings: List of embeddings
        """
        
        logger.info("Generating embeddings for the documents")
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
        
        embeddings = self.embed_documents(documents)
        
        # Create vectors with unique IDs
        vectors = [
            (str(idx), 
             embeddings[idx],
             {"text": doc}) 
            for idx, doc in enumerate(documents)
        ]
        
        vector_count = len(vectors)

        logger.info(f"Inserting {vector_count} vectors into database")
        if vector_count>1000:
            for i in range(0, vector_count, 250):
                self.index.upsert(vectors[i:i+250])
        else:
            # Upsert to Pinecone
            self.index.upsert(vectors)


class LoadData:
    """ This function reads in the data and converts it chunks of text"""
    
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
        for file_path in file_paths:
            for i in self.load_and_preprocess_data(file_path):
                documents.append(i.replace("\n", " "))
            logger.info(f"Data from {file_path} has been processed \n")
            
        return documents


def main():
    load_dotenv()

    cfg = Configurations()
    
    vector_db_pipeline = PopulateVectorDbPipeline(
        pinecone_api_key=cfg.pinecone_api_key,
        pinecone_index_name=cfg.pinecone_index,
    )   
    
    vector_db_pipeline.delete_and_create_db(cfg.delete_and_create_vect_db)    
    load_data = LoadData()

    documents = load_data.process_documents(cfg.filepaths)    
    vector_db_pipeline.upsert_documents(documents)


if __name__ == "__main__":
    main()

