import gradio as gr
from ultralytics import YOLO, SAM
import numpy as np
import cv2
import torch

# Load Models
yolo_model = YOLO("models/best-23.pt")
sam_model = SAM("models/sam2_b.pt")

# Segmentation Function
def segment_tumor(image):
    output_image = image.copy()
    results = yolo_model(image)

    for result in results:
        if result.boxes is not None and result.boxes.cls.numel() > 0:
            bboxes = result.boxes.xyxy.cpu()
            classes = result.boxes.cls.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()

            sam_results = sam_model(
                image,
                bboxes=bboxes,
                verbose=False,
                save=False,
                device=0,
                show=False
            )

            for i, sam_output in enumerate(sam_results):
                if not hasattr(sam_output, "masks") or sam_output.masks is None:
                    continue

                for mask in sam_output.masks.data:
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[:, :, 1] = mask_np
                    output_image = cv2.addWeighted(output_image, 1.0, colored_mask, 0.5, 0)

                x1, y1, x2, y2 = map(int, bboxes[i])
                label = f"{yolo_model.names[int(classes[i])]} {confs[i]:.2f}"
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)
    return output_image

# Launch App
interface = gr.Interface(
    fn=segment_tumor,
    inputs=gr.Image(type="numpy", label="Upload MRI Image"),
    outputs=gr.Image(type="numpy", label="Tumor Segmentation Output"),
    title="🧠 Brain Tumor Detection and Segmentation",
    description="YOLOv11 detects the tumor and SAM2 highlights the shape in MRI images.",
    allow_flagging="never"
)

interface.launch()
# Gradio App will be written here
