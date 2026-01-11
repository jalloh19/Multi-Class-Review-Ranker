# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('words'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"

# Copy the rest of the application
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
