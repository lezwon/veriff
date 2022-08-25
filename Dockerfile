FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app
WORKDIR /app
RUN chmod +x ./start.sh
ENTRYPOINT /app/start.sh
