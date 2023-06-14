FROM tensorflow/tensorflow:2.8.0-gpu
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get --allow-insecure-repositories update && apt-get install gdal-bin=3.0.4+dfsg-1build3 libgdal-dev=3.0.4+dfsg-1build3 libgl1=1.3.2-1~ubuntu0.20.04.2 -y --no-install-recommends
COPY . /GRID
WORKDIR /GRID
RUN pip install --no-cache-dir --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1