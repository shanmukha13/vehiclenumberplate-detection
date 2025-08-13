## Vehicle and Number Plate Detection (YOLOv8 + EasyOCR)

Detect vehicles and read license plates from images or videos using YOLOv8 for detection and EasyOCR for text extraction. Includes a minimal Streamlit web app and an offline pipeline to generate annotated videos.

### Data

The sample video used in this project is available on Pexels: [Traffic flow in the highway](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/).

## Quick start

1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

2) Place model files

- Vehicle model `yolov8n.pt` is downloaded automatically by Ultralytics on first run.
- License plate detector: put your file at `models/license_plate_detector.pt`.
  - Trained with YOLOv8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and the [training tutorial](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide).
  - A pretrained model is available on the author's [Patreon](https://www.patreon.com/ComputerVisionEngineer).

3) Install SORT (for multi-object tracking)

- Download from [abewley/sort](https://github.com/abewley/sort) and copy the `sort` folder into this project root so imports like `from sort.sort import *` work.

## Run the web app (recommended for quick tests)

```bash
streamlit run web_app.py
```

Then open the local URL in your browser and upload an image or a video. If `models/license_plate_detector.pt` is missing, the app will still show vehicle boxes without plate reading.

## Run the offline pipeline (CSV + annotated video)

1) Put an input video at `./sample.mp4` (or edit the path in `main.py`).
2) Run detection and CSV export:

```bash
python main.py
```

This creates `test.csv`.

3) Interpolate missing frames (optional but improves visualization):

```bash
python add_missing_data.py
```

This creates `test_interpolated.csv`.

4) Render the annotated video:

```bash
python visualize.py
```

This creates `out.mp4` in the project root.

## Configuration tips

- EasyOCR runs on CPU by default. To try GPU, change the reader initialization in `util.py` to `gpu=True` if you have a compatible setup.
- Vehicle classes used: `[2, 3, 5, 7]` (car, motorcycle, bus, truck). Adjust in `main.py` or `web_app.py` to your needs.
- For higher accuracy, swap `yolov8n.pt` with a larger YOLOv8 model.

## Project structure

```
.
├─ main.py                  # Batch pipeline: detect, track, OCR → CSV
├─ add_missing_data.py      # Interpolate missing frames in CSV
├─ visualize.py             # Produce annotated video from CSV
├─ web_app.py               # Streamlit web UI (image/video)
├─ util.py                  # OCR utilities and helpers
├─ requirements.txt
├─ LICENSE
└─ models/
   └─ license_plate_detector.pt  # place your plate detector here
```

## License

This repo includes a `LICENSE`. Respect third-party licenses for models and datasets.

## Acknowledgements

- Ultralytics YOLOv8
- EasyOCR
- SORT (by Bewley et al.)
