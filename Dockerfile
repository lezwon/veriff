FROM tensorflow/tensorflow:2.5.0
COPY .requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY . /app
WORKDIR /app
RUN uvicorn main:app --host 0.0.0.0 --port 80
