# **🕹️ Gesture-Controlled Flappy Bird (Deep Learning Edition)**

## **🧠 Abstract**

This project reimagines the **classic Flappy Bird** game with a **modern deep learning twist** — you control the bird using your **hand gestures**\!

Using a **webcam (OpenCV)** for real-time video input, the application employs **MediaPipe** to extract **21 3D hand landmarks** and a **custom-trained Multi-Layer Perceptron (MLP)** model (built with **TensorFlow/Keras**) to classify gestures.

The **game** itself is built in **Pygame**, where each gesture controls the bird’s behavior:

* 👍 **THUMBS\_UP** → Bird flaps (goes up)  
* 👎 **THUMBS\_DOWN** → Bird dives (goes down faster)  
* ✋ **NEUTRAL** → Bird falls normally due to gravity

This repository includes the **complete pipeline** — from data collection and model training to real-time gameplay and model analysis.

## **📁 Project Structure**

OWN-FLAPPY/  
├── \_\_pycache\_\_/  
├── DataCollector.py            \# Records your hand gestures via webcam  
├── GameEnvironmentModule.py    \# Core Pygame logic for Flappy Bird  
├── HandRecognitionModule.py    \# Handles MediaPipe hand tracking & coordinates  
├── GestureInterpreterModule.py \# Loads and runs the trained MLP model  
├── ModelTrainer.py             \# Trains the MLP model on gesture data  
├── ModelAnalyzer.py            \# Analyzes model performance and generates plots  
├── main.py                     \# Main script to run the gesture-controlled game  
├── requirements.txt            \# All necessary Python libraries  
│  
├── (Generated Files)  
├── gestures.csv                \# Collected gesture dataset  
├── gesture\_model.keras         \# Trained MLP model  
├── scaler.pkl                  \# Data scaler used in preprocessing  
├── label\_encoder.pkl           \# Label encoder for gestures  
├── training\_history.pkl      \# Model training history  
├── confusion\_matrix.png        \# Saved confusion matrix plot  
└── training\_history.png         \# Saved accuracy/loss plot

└── shap\_summary\_plot.png      \# Saved accuracy/loss plot

## **⚙️ Setup and Run Instructions**

### **🧩 Prerequisites**

* Python 3.10 (64-bit only)  
  (This project will not work on Python 3.11, 3.12, 3.13, or any 32-bit version of Python)  
* **A webcam**

### **🪜 Step-by-Step Guide**

#### **Step 1: Install Dependencies**

1. Download all .py files and the requirements.txt file into a single folder.  
2. Open your terminal (VS Code Terminal or Command Prompt).  
3. Navigate to your project folder:  
   cd path/to/your/OWN-FLAPPY/folder

4. Install all dependencies using Python 3.10's pip:  
   py \-3.10 \-m pip install \-r requirements.txt

#### **Step 2: Collect Gesture Data**

You must teach the model what your gestures look like.

1. Run the data collector:  
   py \-3.10 DataCollector.py

#### **Step 3: Train Your MLP Model**

Now that you have data, you can train your deep learning model.

1. Run the model trainer:  
   py \-3.10 ModelTrainer.py

2. This script will load gestures.csv, train the MLP for 50 epochs, and save all the necessary files (gesture\_model.keras, scaler.pkl, etc.).

#### **Step 4 (Optional): Analyze Your Model**

Want to see how well your model trained?

1. Run the model analyzer:  
   py \-3.10 ModelAnalyzer.py

2. This will create three image files in your folder: confusion\_matrix.png and training\_history.png,shap\_summary\_plot.png .

#### **Step 5: Play the Game\!**

You're all set\! Your trained model will be loaded automatically.

1. Run the main game script:  
   py \-3.10 main.py

Two windows will open. Use your hand in the "Gesture Control" window to control the bird in the "Gesture Flappy Bird" window. Good luck\!
Report link - https://acrobat.adobe.com/id/urn:aaid:sc:AP:87430acc-c132-4183-9a55-3898f6d5bea8

