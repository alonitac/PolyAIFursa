# YOLO Object Detection Service

This is a FastAPI-based web service that performs object detection on uploaded images using the YOLOv8 model. The application analyzes images, detects objects, and stores prediction results in a SQLite database for later retrieval.

## Setup Instructions

1. Python 3 should be already installed on your Ubuntu  instance. Install some essential packages:

```bash
sudo apt update
sudo apt install python3.14-venv python3-pip libgl1
```

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/alonitac/YoloService.git
cd YoloService
```


1. Create Python virtual environment (venv) and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Python virtual environment is used to isolate project dependencies and ensure that the required packages are installed without affecting the global Python environment.

Your terminal prompt should now indicate that you are in the virtual environment (e.g., `(.venv) ubuntu@hostname:~/$`).

1. Install requirements:
```bash
pip install -r torch-requirements.txt
pip install -r requirements.txt
```

1. Run the application:
```bash
python app.py
```

The service will be available at http://<your_server_ip>:8080

You can test the api endpoints using `curl` or Postman. See the API Endpoints section below for details on available endpoints and how to use them.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence score (0.0–1.0) for a detection to be reported. Raise it to get only high-confidence results; lower it to catch more objects. |

Example:
```bash
export CONFIDENCE_THRESHOLD=0.7
python app.py
```

## Running Tests

The test suite uses `pytest` and FastAPI's built-in test client — no running server needed.

```bash
pytest tests/
```


## API Endpoints

* `POST /predict` - Upload an image for object detection
* `GET /prediction/{uid}` - Get details of a specific prediction by ID
* `GET /predictions/label/{label}` - Get all predictions containing a specific object label (e.g., "person", "car")
* `GET /predictions/score/{min_score}` - Get predictions with confidence score above threshold (e.g., 0.5)
* `GET /prediction/{uid}/image` - Get the processed image with detection boxes
* `GET /image/{type}/{filename}` - Get original or predicted image by filename

## Testing the API

You can use tools like curl, Postman, or a web browser to test the endpoints. For example:

1. Upload an image:
```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:8080/predict
```

2. View detection results (replace {uid} with the ID returned from the upload):
```bash
curl http://localhost:8080/prediction/{uid} 