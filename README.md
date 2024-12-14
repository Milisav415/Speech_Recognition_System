# Speech Recognition System

You can view the main script [here](https://raw.githubusercontent.com/Milisav415/Speech_Recognition_System/refs/heads/main/main.py).

## 📂 Table of Contents

1. [Key Features](#key-features)
2. [How It Works](#how-it-works)
3. [Applications](#applications)
4. [Installation](#installation)

## 🚀 Description

The **Speech Recognition System** is a machine learning-based solution designed to recognize spoken words from audio input. It trains a model using a provided audio database, allowing for customization and high accuracy in specific applications. The system is capable of processing diverse speech patterns, accents, and background noise, making it versatile and effective.

## Key Features:
- **Custom Model Training**: Trains a word recognition model based on your database of audio samples.
- **High Accuracy**: Optimized algorithms ensure precise word recognition even in noisy environments.
- **Scalability**: Easily extend the system to recognize additional words or phrases by updating the training database.
- **Efficiency**: Produces a trained model ready for deployment in minimal time.

## How It Works:
1. **Database Input**: 
   - Provide a curated database of audio samples labeled with corresponding words.
   - This system is designed around four different words, or "classes," as they are referred to in the source code. The number of classes is modular and easy to change by editing a constant in the source code.
   - The database must have the same number of examples (audio files) for each word.
   - Finaly the algorithm goes through each audio file and extractes the features from them, down below you can see MFCCs through time.
   <img src="Speech_Recognition_System/assets/log_mel_spec.PNG" width="256" height="256">
   
2. **Model Training**: 
   - The system processes the database to train a model tailored to the input data, by reducing the dimensions of the feature vector down to 3D space, here is picture of the final result:
   - <img src="assets/2024-12-14_16-20.PNG" alt="Alt Text" width="256" height="256">
   -here we can see all 4 calsses nicely gruped up and ready for classification.

3. **Recognition**: 
   - The trained model recognizes spoken words from new audio inputs, by using the SVM method of training a classifier.
   - With this, we get a classifier that will perfectly position itself in the middle between each pair of classes.
   <img src="assets/t_cm.PNG" alt="Alt Text" width="256" height="256"> <img src="assets/tr_cm.PNG" alt="Alt Text" width="256" height="256">

## Applications:
- **Speech-to-Text Systems**: Convert spoken words into text for documentation or communication tools.
- **Voice-Activated Interfaces**: Enable smart devices or applications to respond to user commands.
- **Language Processing**: Create tools for linguistic analysis or language learning.
- **Customizable Solutions**: Adapt the system for industry-specific requirements, such as healthcare or education.

---

## ⚙️ Installation

Step-by-step guide to set up your project locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/Milisav415/Speech_Recognition_System.git
2. Add the path to your test and training database in the main section

## **Thank you for exploring my project!** 
If you'd like to learn more about my background and qualifications, please visit my [LinkedIn profile](https://www.linkedin.com/in/milisav-jovanovic-059969336/)
