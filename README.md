# Research Paper Clustering Visualization

This project provides an interactive visualization of research paper clusters using FastAPI backend and vanilla JavaScript frontend. It uses HDBSCAN and K-means clustering algorithms along with SPECTER2 embeddings to group similar research papers.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository
`git clone [your-repository-url]`
`cd [repository-name]`

2. Create and activate a virtual environment
On macOS/Linux:
`python3 -m venv venv`
`source venv/bin/activate`

On Windows:
`python -m venv venv`
`.\venv\Scripts\activate`

3. Install required packages
`pip install fastapi uvicorn sentence-transformers transformers torch scikit-learn hdbscan`

## Running the Application

1. Start the FastAPI backend server (from the project root directory)
Make sure your virtual environment is activated
`uvicorn server:app --reload --port 8000`

2. Start the frontend server (from the static directory in a new terminal)
Navigate to the static directory
`cd static`

Start Python's built-in HTTP server
`python -m http.server 3000`

3. Access the application
Open your web browser and navigate to:
`http://localhost:3000`

## Test out the system

You could input any query related to a research topic ranging from a few words to a drafted abstract. We have used such as "virtual reality walking" when testing the system.

