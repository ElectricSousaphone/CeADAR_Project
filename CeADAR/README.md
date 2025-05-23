# CeADAR_Project

## RAG Model Pipeline
### Overview
This pipeline reads documents from the Input_documentation folder, populates a vector database, processes queries, 
and returns answers on an app.

### Requirements

Python 3.8+

huggingface_hub

langchain_text_splitters

loguru

pinecone

PyPDF2

python-dotenv

python_docx

streamlit


### Installation
Clone the repository:

git clone https://github.com/ElectricSousaphone/CeADAR_Project.git

cd CeADAR_Project

Install the required libraries:

pip install -r requirements.txt

Create a .env file with the variables "PINECONE_API_KEY" this can be found when creating a pinecone vector database. "HF_TOKEN" this can be found when creating a hugging face account.


### Usage

Step 1: Populate Vector Database

Place your documents in the Input_documents folder.

Run the vector_db_data_pipeline.py script to populate the vector database.


Step 2: Query and Answer

In the command line enter 'streamlit run front_end.py'

Open your browser and navigate to http://localhost:8501.

Enter your query in the dashboard and get the answer.

### File Structure
Input_documentation/: Folder containing input documents.

vector_db_data_pipeline.py: Script to preprocess data and populate the vector database.

answer_retrieval.py: Script that 

front_end.py: Streamlit application to handle queries and display answers.

requirements.txt: List of required libraries.

Documentation/: Folder containing the architecture diagrams 


### Contributing

Feel free to submit issues and pull requests to improve the pipeline.

### License

None