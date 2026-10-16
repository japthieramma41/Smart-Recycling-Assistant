py train_model.py --helppy train_model.py --data_dir dataset-resized --output_dir models --epochs 10# AI-Powered Smart Recycling Assistant

A beginner-friendly hackathon prototype for classifying waste images and displaying a simulated Digital Twin of waste bins.

## 1. Dataset

Use the public TrashNet repository:
https://github.com/garythung/trashnet

Download the ZIP and extract it, or clone it:
```bash
git clone https://github.com/garythung/trashnet.git
```
The training script searches the supplied folder for six class folders: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`. In the commonly used repository, these are available under `dataset-resized`.

## 2. Set up Python

Python 3.10 or 3.11 is a good starting point for TensorFlow compatibility.

Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Train the model

From this project folder, run:
```bash
python train_model.py --data_dir "C:/path/to/trashnet" --epochs 8
```
Replace the path with the actual extracted dataset location. If the class folders are in `trashnet/dataset-resized`, you can pass either the repository root or that folder.

The script:
- Finds the six class folders.
- Splits data into training, validation, and test sets.
- Trains a MobileNetV2 transfer-learning classifier.
- Prints held-out test accuracy, a classification report, macro F1-score, and confusion matrix.
- Saves `models/waste_classifier.keras`, `models/class_names.json`, and `models/evaluation.json`.

The first run may download pre-trained MobileNetV2 weights and therefore needs an internet connection. Training time depends on hardware. Google Colab with a GPU can be used if local training is slow.

## 4. Run the application

After the model files are saved in the `models` folder:
```bash
streamlit run app.py
```
Open the local URL printed by Streamlit in your browser. Upload an image, click **Classify Waste**, review the result, then click **Confirm classification and update virtual bin** to record it.

## 5. What the prototype does

- Predicts one of six TrashNet classes.
- Displays a category-specific disposal tip.
- Stores confirmed classifications in `data/classification_records.csv`.
- Displays category counts and simulated virtual bin fill levels.
- Allows downloading the records as a CSV file.

## 6. Important limitations

- The model always selects one of its six trained categories, even for an unrelated image. Confidence is not a guarantee of correctness.
- TrashNet is limited and may not represent campus conditions. Test with separate campus photographs and report the results honestly.
- Disposal guidance must follow local waste collection rules.
- Virtual bin fill percentages are simulated using assumed increments per item. They are not physical measurements. Real fill levels require sensors or manual measurements.
- The CSV is local prototype storage and may be reset if the app runs in an ephemeral hosting environment.

## 7. Suggested two-day priorities

Day 1: Dataset, model training, evaluation, and basic upload/prediction interface.
Day 2: Disposal guidance, virtual bins, charts, end-to-end testing, and final demo.
