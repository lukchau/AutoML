FROM python:3.10

WORKDIR / .

COPY requirements.txt ./requirements.txt

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV API_PORT=$API_PORT
ENV HOST=$HOST

#RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt && \
    pip uninstall -y bcrypt && \
    pip install --no-cache-dir --force-reinstall bcrypt

COPY . .

CMD gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind $HOST:$API_PORT