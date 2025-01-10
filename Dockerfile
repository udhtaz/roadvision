FROM python:3.11.10-slim-bookworm

RUN apt-get update -y --no-install-recommends

# Install linux packages, gcc compiler and opencv prerequisites
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
# pkg-config and libhdf5-dev (not included) are needed to build 'h5py==3.11.0' aarch64 wheel required by 'tensorflow'
RUN apt-get update -y --no-install-recommends && \
	apt-get install -y --no-install-recommends \
	nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
	python3-opencv ca-certificates python3-dev wget sudo curl unzip cmake ninja-build \
	python3-pip zip htop gcc libgl1 libpython3-dev gnupg g++ libusb-1.0-0 \
	&& rm -rf /var/lib/apt/lists/*

# Detectron2 prerequisites
RUN pip install torch==2.5.0 torchvision==0.20.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir --default-timeout=5000
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --default-timeout=5000


RUN git config --global http.postBuffer 1048576000
RUN git config --global https.postBuffer 1048576000
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6' --default-timeout=5000

WORKDIR /app

COPY requirements.txt /app

RUN pip3 install -r requirements.txt

COPY . /app

RUN pip install -r requirements.txt --default-timeout=5000 --no-cache-dir

EXPOSE 80

CMD ["python", "app.py"]