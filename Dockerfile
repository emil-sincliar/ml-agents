FROM python:3.9-slim

WORKDIR /app

# Copy the model files and application
COPY . .

# Install dependencies
RUN pip install -e .
RUN pip install streamlit

# Expose the Streamlit port
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
