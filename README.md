# RoadVision

RoadVision is a Flask-based computer vision application designed to detect surface irregularities such as bumps and potholes on roadways. By leveraging cutting-edge deep learning frameworks like Detectron2 and YOLO, ROADVISION provides reliable and efficient analysis of road conditions through image and video segmentation.

https://github.com/user-attachments/assets/cb679542-2aa1-4542-8d63-ba48783c27be

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [Endpoints](#endpoints)
- [Environment Variables](#environment-variables)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Multi-Model Support**: Integrates both Detectron2 and YOLO for object detection.
- **Blurry Image Detection**: Automatically filters blurry images to ensure accurate analysis.
- **Video Processing**: Processes video files or live camera streams for segmentation.
- **Extensible Configuration**: Highly configurable models and rules for different use cases.
- **RESTful API**: Fully functional API for image and video analysis.
- **Dockerized Deployment**: Easily deployable using Docker and a Bash script.

---

## Project Structure

The repository is structured as follows:

```plaintext
roadvision/
├── inference/                # Image inference foerm model predictions
├── training_scripts/         # Scripts to train custom models
├── app.py                    # Main Flask application
├── config.py                 # Configuration for models and rules
├── helpers.py                # Helper functions for preprocessing and inference
├── init.sh                   # Script to build and run Docker container
├── rules.py                  # Business rules for image analysis
├── requirements.txt          # Python dependencies
├── roadvisionDetectron2.pth  # Custom-trained weights for Detectron2
├── roadvisionYOLO11.pt       # Custom-trained weights for YOLOv11
├── Dockerfile                # Dockerfile for building the container
├── roadvisionSheet.xlsx      # Spreadsheet defining detection classes
├── LICENSE                   # Licensing information
└── README.md                 # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.11
- Docker
- NVIDIA CUDA (optional for GPU acceleration)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/udhtaz/roadvision.git
   cd roadvision
   ```

2. Install dependencies (Optional, automatically installs when building docker):

   ```bash
   pip install -r requirements.txt
   ```

3. Build and run the Docker container using the Bash script:

   ```bash
   bash init.sh
   ```

---

## Usage

### Running the API

To start the ROADVISION API, execute the following command using Docker and the provided Bash script:

```bash
bash init.sh
```

### Endpoints

- **API Status**

  - **GET** `/`\
    Check if the API is running.

- **Image Segmentation**

  - **POST** `/analyse_image_segm/<roadvision_channel>`\
    Upload an image for segmentation. Specify `roadvision_channel` as `Detectron2` or `YOLO11`.

- **Video Segmentation**

  - **POST** `/vid_segm/<roadvision_channel>`\
    Upload a video for segmentation. Specify `roadvision_channel` as `Detectron2` or `YOLO11`.

---

## Environment Variables

Add a `.env` file in the root directory to configure environment variables:

```plaintext
PORT=80
DEBUG=False
```

---

## References

The ROADVISION project utilizes datasets from the following sources:

1. **Pothole Dataset 1**:

   - Chitale, P. A., Kekre, K. Y., Shenai, H. R., Karani, R., & Gala, J. P. (2020). Pothole Detection and Dimension Estimation System using Deep Learning (YOLO) and Image Processing. *2020 35th International Conference on Image and Vision Computing New Zealand (IVCNZ)*, 1-6. doi: [10.1109/IVCNZ51579.2020.9290547](https://doi.org/10.1109/IVCNZ51579.2020.9290547)
GitHub Repository: [https://github.com/jaygala24/pothole-detection](https://github.com/jaygala24/pothole-detection)

2. **Pothole Dataset 2**:

   - Dataset Ninja. (2024). Visualization Tools for Road Pothole Images Dataset. Retrieved December 2, 2024, from [https://datasetninja.com/road-pothole-images](https://datasetninja.com/road-pothole-images)

3. **Speed Bump Dataset**:

   - VARMA, V. S. K. P. (2018). Speed Hump/Bump Dataset. *Mendeley Data*, V1. doi: [10.17632/xt5bjdhy5g.1](https://data.mendeley.com/datasets/xt5bjdhy5g/1)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m 'Add a feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

---

