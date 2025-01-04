FROM python:3.8-slim-buster
# FROM python:3.9

RUN apt-get update -y --no-install-recommends

# gcc compiler and opencv prerequisites
RUN apt-get -y --no-install-recommends install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
	rm -rf /var/lib/apt/lists/*

# Detectron2 prerequisites
RUN pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html --no-cache-dir
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2 --no-cache-dir
# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /app


COPY requirements.txt /app

RUN pip3 install -r requirements.txt

# COPY . /app
COPY *.py *.xlsx *.pth *.env /app/

EXPOSE 80

CMD python app.py