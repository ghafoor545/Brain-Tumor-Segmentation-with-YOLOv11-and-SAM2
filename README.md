# 🧠 Brain Tumor Segmentation with YOLOv11 and SAM2

This project is a deep learning-based web app that detects and segments brain tumors in MRI scans using **YOLOv11** for object detection and **SAM2 (Segment Anything Model)** for precise tumor shape segmentation. The app is built using **Gradio** for a simple and interactive web interface.

![Instant Segmentation Preview](assets/Instant%20Segmentaion.jpg)

---

## 🚀 Features

- ✅ Detects **Glioma**, **Meningioma**, and **Pituitary tumors**
- ✅ Uses **YOLOv11** for accurate bounding box detection
- ✅ Uses **SAM2** for high-resolution shape segmentation
- ✅ Web interface with **Gradio**
- ✅ Easy to install and run locally

---

## 📁 Project Structure
brain-tumor-segmentation/
├── app/
│   └── app.py               # Gradio web app
├── models/
│   ├── best.pt              # YOLOv11 trained weights
│   └── sam2_b.pt            # SAM2 segmentation model
├── assets/
│   ├── sample_input.jpg     # Example MRI input
│   └── sample_output.jpg    # Example segmented output
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git exclusions



---

## 🧰 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ghafoor545/Brain-Tumor-Segmentation-with-YOLOv11-and-SAM2.git
cd Brain-Tumor-Segmentation-with-YOLOv11-and-SAM2
2. Create and Activate Virtual Environment (optional)
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Web App

Place the trained models in the models/ folder:

models/best.pt — YOLOv11 weights
models/sam2_b.pt — SAM2 weights
Then launch the app:

cd app
python app.py
Gradio will provide a local URL like:

Running on http://127.0.0.1:7860/
````



## 📦 Models Used

Ultralytics YOLOv11
Meta AI Segment Anything (SAM2)
Roboflow for dataset management



## 🙌 Acknowledgments

YOLOv11 by Ultralytics
SAM2 by Meta AI
Roboflow for providing annotated medical datasets

## Ghafoor Khan
