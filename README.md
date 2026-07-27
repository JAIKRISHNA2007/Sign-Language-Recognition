# 🤟 Sign Language Recognition using MediaPipe & Random Forest

A real-time **American Sign Language (ASL)** recognition system built using **MediaPipe**, **Random Forest**, **OpenCV**, and **Streamlit**. The application detects hand landmarks from a webcam feed, predicts ASL alphabet gestures in real time, displays prediction confidence, and allows users to build complete words interactively.

---

# 🚀 Live Demo

### 🌐 Live Application

https://sign-language-recognition-version1.streamlit.app/

### 📂 GitHub Repository

https://github.com/JAIKRISHNA2007/Sign-Language-Recognition

### 🎥 Project Demonstration

https://drive.google.com/file/d/1CIfvCqkTBoVWGJuz7J8rJSHUccYhfNu_/view?usp=drive_link

### 📥 Trained Model Download

https://drive.google.com/drive/folders/1rnF8viszOJfsc73_ETXkF1SqFgCkrDZz?usp=drive_link

---

# 📌 Features

- ✅ Real-time webcam-based ASL recognition
- ✅ MediaPipe 21-hand landmark extraction
- ✅ Random Forest Machine Learning classifier
- ✅ Live confidence score
- ✅ Automatic word builder
- ✅ Space gesture support
- ✅ Delete gesture support
- ✅ Stable prediction filtering
- ✅ Interactive Streamlit web interface
- ✅ Automatic model download support
- ✅ Easily deployable on Streamlit Community Cloud

---

# 🛠 Tech Stack

- Python
- Streamlit
- MediaPipe
- OpenCV
- Scikit-learn
- Random Forest
- NumPy
- Joblib

---

# 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── models/
│
├── src/
│   ├── extract_landmarks.py
│   ├── train_model.py
│   └── predict_webcam.py
│
├── screenshots/
│   ├── home.png
│   ├── letter_prediction.png
│   └── word_builder.png
│
└── data/
```

---

# ⚙️ Project Workflow

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
63 Numerical Features
          │
          ▼
Random Forest Model
          │
          ▼
Real-Time Prediction
          │
          ▼
Word Builder
```

---

# 📸 Screenshots

## 🏠 Home Screen

![Home](screenshots/home.png)

---

## ✋ Letter Prediction

![Prediction](screenshots/letter_prediction.png)

---

## 📝 Word Builder

![Word Builder](screenshots/word_builder.png)

---

# 📦 Installation

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

Download the **ASL Alphabet Dataset** from:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Extract the dataset into

```text
data/raw/asl_alphabet_train/
```

Generate landmark features

```bash
python src/extract_landmarks.py
```

Train the Random Forest model

```bash
python src/train_model.py
```

Run the application

```bash
streamlit run app.py
```

---

# 🧠 Machine Learning Model

Algorithm:

- Random Forest Classifier

Input:

- 21 MediaPipe Hand Landmarks
- 63 Numerical Features

Output:

- ASL Alphabet Prediction

---

# 📁 Dataset

Dataset Used:

**ASL Alphabet Dataset**

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

# 📥 Model

The trained model is hosted separately because GitHub has a 100 MB file size limit.

Download it from:

https://drive.google.com/drive/folders/1rnF8viszOJfsc73_ETXkF1SqFgCkrDZz?usp=drive_link

Place it inside:

```text
models/sign_model.pkl
```

---

# 🔮 Future Improvements (Version 2)

- Deep Learning-based recognition (LSTM / Transformer)
- Dynamic sign recognition
- Sentence generation
- Text-to-Speech
- Speech-to-Text
- Support for additional sign languages
- Higher prediction accuracy
- Better UI/UX
- Faster inference
- Mobile application support

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**JAI KRISHNA S**

Computer Science Engineering Student

---

# 🏢 Internship Information

**Organization:** CodTech IT Solutions

**Internship:** AI Internship

**Project:** Sign Language Recognition using MediaPipe & Random Forest

**Intern ID:** CTTS162

---

⭐ If you found this project helpful, please consider giving it a **Star ⭐** on GitHub.