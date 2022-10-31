FROM tensorflow/tensorflow:2.5.0-gpu

WORKDIR /c/Users
RUN apt update -y
RUN apt install -y libgl1-mesa-glx
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY figout/src .
COPY collect_data.py .
COPY test_model.py .
COPY resources/healthy900_test.tfrecords ./resources/healthy900_test.tfrecords
COPY resources/healthy900_train.tfrecords ./resources/healthy900_train.tfrecords

ENTRYPOINT ["python3", "-m", "test_model"]