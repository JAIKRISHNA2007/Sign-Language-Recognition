# 🤟 Sign Language Recognition using MediaPipe and Random Forest

A real-time Sign Language Recognition system that recognizes American Sign Language (ASL) alphabet gestures using **MediaPipe Hand Landmarks** and a **Random Forest Machine Learning model**. The application provides live webcam prediction through a Streamlit web interface and automatically forms words from recognized gestures.

---

## Features

- Real-time webcam sign language recognition
- MediaPipe hand landmark detection
- Random Forest classifier
- Prediction confidence display
- Automatic word formation
- Support for **Space** and **Delete** gestures
- Interactive Streamlit web application

---

## Project Workflow

```
ASL Alphabet Dataset
        │
        ▼
MediaPipe Hand Detection
        │
        ▼
21 Hand Landmarks
        │
        ▼
63 Landmark Features
        │
        ▼
Random Forest Classifier
        │
        ▼
Real-Time Sign Prediction
```

---

## Technologies Used

- Python
- Streamlit
- Streamlit WebRTC
- MediaPipe
- OpenCV
- NumPy
- Scikit-Learn
- Joblib

---

## Project Structure

```text
Sign-Language-Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   ├── extract_landmarks.py
│   ├── train_model.py
│   └── predict_webcam.py
│
├── screenshots/
│
├── data/
│   ├── raw/
│   └── processed/
│
└── models/
```

---

## Dataset

This project uses the **ASL Alphabet Dataset**.

Dataset Source:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The original dataset is **not included** in this repository because of its size.

To generate the processed landmark dataset:

```bash
python src/extract_landmarks.py
```

---

## Model

The trained model (`sign_model.pkl`) is **not included** because it exceeds GitHub's file size limit.

To train the model:

```bash
python src/train_model.py
```

The trained model will automatically be saved inside the `models/` directory.

---

## Installation

Clone the repository

```bash
git clone https://github.com/JAIKRISHNA2007/Sign-Language-Recognition.git
```

Move into the project folder

```bash
cd Sign-Language-Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Usage

1. Launch the Streamlit application.
2. Start the webcam.
3. Show an ASL alphabet gesture.
4. View the predicted letter and confidence score.
5. Hold a gesture to automatically build words.

---

## Screenshots

*Screenshots will be added after deployment.*

---

## Demo Video

*A demo video will be added after deployment.*

---

## Future Improvements

- Improve prediction accuracy
- Deep Learning based classification
- Sentence generation
- Text-to-Speech conversion
- Mobile application
- Multi-hand recognition

---

## Internship Information

This project was developed as part of the **AI Internship** at **Codtech IT Solutions Private Limited**.

- **Intern ID:** CTTS162
- **Intern:** JAI KRISHNA S

---

## Author

**JAI KRISHNA S**

---

## License

This project is licensed under the **MIT License**.