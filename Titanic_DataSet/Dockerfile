FROM jupyter/datascience-notebook:latest 

WORKDIR /home/jovyan/work 
COPY . /home/jovyan/work/ 
EXPOSE 8888
CMD ["start-notebook.sh","--NotebookApp.token=''"]
# docker build -t titanic_project . 
#                 -Container name

 # docker run -p 8888:8888 titanic_project
 #                         -Conttainer name

 # template:
# docker build -t my-app . 

# docker run -d -p 8080:80 --name my-app-container my-app