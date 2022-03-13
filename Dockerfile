FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY ./app /app

RUN pip install --trusted-host pypi.python.org flask
RUN pip install --trusted-host pypi.python.org torch
RUN pip install --trusted-host pypi.python.org torchvision
RUN pip install --trusted-host pypi.python.org torchaudio
