# Chainlit Application Documentation

## Overview
This documentation provides an overview of the Chainlit application for the Retrieval-Augmented Generation (RAG) functionality using PubMed data. It outlines the setup instructions, usage guidelines, and key components of the application.

## Setup Instructions

1. **Clone the Repository**
   Clone the repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Install Dependencies**
   Navigate to the project directory and install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory based on the `.env.example` file. Ensure to fill in the necessary environment variables such as API keys and configuration settings.

4. **Data Preparation**
   Place your PubMed CSV data file in the `data` directory. Ensure that the file is named `cleaned_pubmed_papers.csv` for the application to load it correctly.

5. **Run the Application**
   Start the Chainlit application by running:
   ```
   chainlit run app.py
   ```

## Usage Guidelines

- Once the application is running, you can access it via the provided local URL.
- Input your questions related to PubMed papers in the designated input field.
- The application will retrieve relevant documents and generate responses based on the provided queries.

## Key Components

- **app.py**: Entry point for the Chainlit application, initializing routes and configurations.
- **src/rag**: Contains modules for embeddings, vector store management, and prompt templates.
- **src/models**: Implements the logic for interacting with the language model.
- **src/utils**: Provides utility functions for data loading and processing.

## Contribution
For contributions, please fork the repository and submit a pull request with your changes. Ensure to follow the coding standards and include tests for new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.