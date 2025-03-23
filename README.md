# subtitle_search_engine
# Subtitle Search Engine (Cloning Shazam)

## Background
In the fast-evolving landscape of digital content, effective search engines play a pivotal role in connecting users with relevant information. This project focuses on improving the search relevance for video subtitles, enhancing the accessibility of video content by leveraging **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.

## Objective
Develop an advanced search engine algorithm that efficiently retrieves subtitles based on user queries, with a specific emphasis on subtitle content. The goal is to enhance the relevance and accuracy of search results using **semantic search techniques**.

## Keyword-Based vs. Semantic Search Engines
- **Keyword-Based Search Engine**: Relies heavily on exact keyword matches between the user query and the indexed documents.
- **Semantic Search Engine**: Goes beyond simple keyword matching to understand the meaning and context of user queries and documents.
- **Comparison**: While keyword-based search engines focus on matching exact words, semantic search engines understand deeper meaning and context, providing more relevant results.

## Core Logic
To compare a user query against a video subtitle document, the core logic involves three key steps:
### 1. Preprocessing of Data
- If compute resources are limited, take a random 10-30% of the data.
- Clean the data by removing timestamps and unnecessary metadata.
- Convert subtitle documents into vector representations.
- Convert user queries into vector representations.

### 2. Cosine Similarity Calculation
- Compute cosine similarity between the document vectors and the user query vector.
- The similarity score determines the relevance of the documents.
- Return the most similar documents.

## Data
Click here to download the subtitle data.

## Step-by-Step Process
### Part 1: Ingesting Documents
1. Read and analyze the given subtitle dataset.
2. Identify that the given data is in a database file format.
3. Review the **README.txt** to understand the structure of the database.
4. Decode and extract subtitle text from the database.
5. If compute resources are limited, randomly sample **30%** of the data.
6. Apply appropriate cleaning steps (e.g., remove timestamps, special characters, stop words).
7. Experiment with different text vectorization techniques:
   - **BOW / TF-IDF** for keyword-based search.
   - **BERT-based SentenceTransformers** for semantic search.
8. **Document Chunking:**
   - Large documents should be divided into smaller chunks to avoid information loss.
   - Set a **token window size** (e.g., 500 tokens per chunk).
   - Use **overlapping windows** to ensure contextual continuity across chunks.
9. Store **vector embeddings** in **ChromaDB**.

### Part 2: Retrieving Documents
1. **User Query Input:**
   - Accept user queries in **audio format** (e.g., a 2-minute clip from a TV series or movie in the database).
2. **Preprocessing:**
   - Convert the audio query into text using a **speech-to-text model**.
   - Clean and normalize the text query.
3. **Query Embedding:**
   - Generate an embedding for the user query using **SentenceTransformers**.
4. **Cosine Similarity Calculation:**
   - Compute similarity scores between the query embedding and subtitle document embeddings stored in ChromaDB.
5. **Return Relevant Results:**
   - Retrieve and rank the most relevant subtitle segments based on cosine similarity scores.

## Tech Stack
- **Python**
- **LangChain**
- **ChromaDB**
- **TF-IDF / BOW / SentenceTransformers**
- **Google Generative AI (Gemini)**
- **Streamlit**
- **Speech-to-Text APIs**

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/subtitle-search-engine.git
cd subtitle-search-engine

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Preprocess Subtitle Files
```bash
python preprocess.py --input subtitles_folder --output db_folder
```

### 2. Start the Search Engine
```bash
streamlit run app.py
```

### 3. Perform a Search
- Enter an **audio query** in the search box.
- The system retrieves relevant subtitle segments based on embeddings and similarity search.
- Results are displayed with timestamps and context.

## Future Enhancements
- Support for **multiple languages**.
- Integration with **real-time video search**.
- Enhanced **speech-to-text models**.
- Improved **ranking algorithms**.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
MIT License

## Contact
For queries, reach out to **your.email@example.com**.
