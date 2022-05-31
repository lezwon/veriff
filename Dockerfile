FROM python:3.7

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app
RUN chmod +x ./start.sh
ENTRYPOINT /app/start.sh
