# PubMed RAG Chainlit

This project implements a Retrieval-Augmented Generation (RAG) application using Chainlit, designed to interact with PubMed research papers. The application leverages embeddings and a vector store to provide intelligent responses to user queries based on the content of the papers.

## Overview

The PubMed RAG Chainlit application allows users to ask questions related to PubMed articles, and it retrieves relevant information from a vector store of embeddings created from the articles' titles and abstracts. The application utilizes the Ollama language model for generating responses.

## Project Structure

```
pubmed-rag-chainlit
├── app.py                  # Entry point for the Chainlit application
├── src                     # Source code for the application
│   ├── rag                 # RAG module
│   │   ├── __init__.py     # RAG package initialization
│   │   ├── embeddings.py    # Embedding logic
│   │   ├── vectorstore.py   # Vector store management
│   │   └── prompts.py       # Prompt templates
│   ├── models              # Models module
│   │   ├── __init__.py     # Models package initialization
│   │   └── llm.py          # LLM logic
│   └── utils               # Utility functions
│       ├── __init__.py     # Utils package initialization
│       └── data_loader.py   # Data loading and processing
├── data                    # Directory for data files
│   └── .gitkeep            # Keeps the data directory in version control
├── vectors                 # Directory for vector files
│   └── .gitkeep            # Keeps the vectors directory in version control
├── chainlit.md            # Chainlit application documentation
├── .env.example            # Example environment variables
├── requirements.txt        # Project dependencies
├── config.json             # Configuration settings
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd pubmed-rag-chainlit
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying `.env.example` to `.env` and updating the values as needed.

## Usage

To run the application, execute the following command:
```
chainlit run app.py
```

Once the application is running, you can access it in your web browser and start asking questions related to PubMed articles.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.