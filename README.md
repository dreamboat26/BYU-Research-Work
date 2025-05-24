# 🧬 BYU Research Work Locating Bacterial Flagellar Motors – YOLOv8 Implementation

This repository contains my research work carried out at BYU, it is an end-to-end pipeline implemenatation which detects and localize bacterial flagellar motors in 3D cryo-electron tomography (cryo-ET) reconstructions.

Leveraging the YOLOv8 object detection framework, this project encompasses data parsing, visualization, model training, evaluation, and submission preparation.

---

## 📁 Project Structure

- `parse data.ipynb`: Processes raw `.mrc` files and accompanying annotations to generate 2D image slices with corresponding bounding boxes suitable for YOLOv8 training.
- `visualizing the dataset.ipynb`: Provides exploratory data analysis (EDA) and visualization of the dataset, including sample images and bounding box distributions.
- `training yolov8.ipynb`: Details the training pipeline for YOLOv8, including data augmentation, hyperparameter tuning, and model checkpointing.
- `reverse engineering yolov8.ipynb`: Analyzes the YOLOv8 architecture and training process to understand model decisions and performance.
- `submission notebook.ipynb`: Generates predictions on the test set and formats them according to the competition's submission requirements.
- `final merged notebook.ipynb`: Consolidates all steps into a single, streamlined pipeline for reproducibility and ease of use.

---

## 🧠 Approach Overview

1. **Data Preparation**: Converted 3D `.mrc` volumes into 2D slices, aligning annotations to create YOLOv8-compatible datasets.
2. **Exploratory Data Analysis**: Visualized sample images and bounding box distributions to inform preprocessing and augmentation strategies.
3. **Model Training**: Trained YOLOv8 models with customized data augmentation techniques and hyperparameter tuning to optimize detection performance.
4. **Model Evaluation**: Assessed model performance using metrics such as mean Average Precision (mAP) and Intersection over Union (IoU).
5. **Submission Generation**: Formatted model predictions to meet competition submission guidelines.

---

## 🧰 Dependencies

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- [mrcfile](https://github.com/ccpem/mrcfile)

---

## Getting Started

### Clone the repository:
```
git clone https://github.com/dreamboat26/BYU-Research-Work.git
cd BYU-Research-Work
```
### Set up the dataset

- Placed the data in the `data/` directory.

### Run data parsing

- Executed `parse data.ipynb` to process raw data into YOLOv8-compatible format.

### Visualize the dataset

- Opened `visualizing the dataset.ipynb` to explore data distributions and sample images.

### Train the model

- Ran `training yolov8.ipynb` to train the YOLOv8 model.

### Evaluate and analyze the model

- Used `reverse engineering yolov8.ipynb` to delve into model performance and architecture.

### Generate submission

- Executed `submission notebook.ipynb` to create a submission file for the competition.

### Run the complete pipeline

- For a streamlined process, ran `final merged notebook.ipynb`.

---

## 📊 Results

- Achieved a validation **mAP of 0.78**

Further improvements are being explored through advanced data augmentation and model ensembling techniques.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- **Brigham Young University** and the competition organizers for providing the dataset and platform.
- The **[Ultralytics](https://github.com/ultralytics/ultralytics)** team for developing and maintaining YOLOv8.
