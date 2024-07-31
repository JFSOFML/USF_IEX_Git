# IMAGE: Use the latest Python 3.12 slim image
FROM python:3.12-slim

# SET: the working directory in the container
WORKDIR /app

# COPY:  the current directory contents into the container at /app
COPY . /app

# INSTALL:  any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE: the Streamlit PORT
EXPOSE 8501

# COMMAND: to run the Streamlit app
CMD ["streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
