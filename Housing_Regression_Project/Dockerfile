# FROM jupyter/datascience-notebook:latest 

# WORKDIR /home/jovyan/work 
# COPY . /home/jovyan/work/ 
# EXPOSE 8888
# CMD ["start-notebook.sh","--NotebookApp.token=''"]

# Base image
FROM python:3.8

# Working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t housing_project . 
#                 -Container name

 # docker run -p 8888:8888 ames_housing_project
 #                         -Conttainer name

 # template:
# docker build -t my-app . 

# docker run -d -p 8080:80 --name my-app-container my-app