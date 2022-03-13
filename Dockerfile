FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY ./app /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
