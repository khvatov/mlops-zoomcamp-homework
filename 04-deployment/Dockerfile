FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ["homework_docker.py", "Pipfile", "Pipfile.lock", "./"]
RUN pip install -U pip
RUN pip install pipenv
RUN pipenv install --system --deploy

CMD [ "python", "homework_docker.py", "2023", "5" ]