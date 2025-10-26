Gesture-Controlled Flappy Bird with Deep LearningAbstractThis project implements the classic Flappy Bird game, but with a modern twist: you control the bird using your hand gestures. The application uses a webcam (OpenCV) to capture your hand, a real-time hand-tracking model (MediaPipe) to extract 21 3D landmarks, and a custom-trained Multi-Layer Perceptron (MLP) deep learning model (built with TensorFlow/Keras) to classify your gestures.The game itself is built in Pygame. The model is trained to recognize three distinct gestures:THUMBS_UP: Makes the bird flap (go up).THUMBS_DOWN: Makes the bird dive (go down faster).NEUTRAL: The bird falls normally due to gravity.This repository includes the complete pipeline: data collection, model training, real-time game implementation, and model analysis.Project StructureOWN-FLAPPY/
├── ─ __pycache__/
├── ─ DataCollector.py         # Script to record your hand gestures
├── ─ GameEnvironmentModule.py # The Pygame flappy bird game logic
├── ─ HandRecognitionModule.py # Handles MediaPipe and calculates relative coordinates
├── ─ GestureInterpreterModule.py # Loads and runs the trained MLP model
├── ─ ModelTrainer.py          # Script to train the MLP model on your data
├── ─ ModelAnalyzer.py         # Script to analyze the model and create plots
├── ─ main.py                  # Main script to run the final game
├── ─ requirements.txt         # All necessary Python libraries
├── ─ gestures.csv             # Your collected gesture data
├── ─ gesture_model.keras      # Your trained MLP model
├── ─ scaler.pkl               # The data scaler for your model
├── ─ label_encoder.pkl        # The label encoder for your model
└── ─ training_history.pkl     # The training history for analysis
Setup and Run InstructionsFollow these steps exactly to set up and run the project.PrerequisitesA 64-bit version of Python, such as Python 3.10 or 3.11. (This project will not work on 32-bit Python or versions 3.12+).A webcam.Step 1: Install DependenciesDownload all the .py files and the requirements.txt file into a single folder.Open your terminal (like VS Code Terminal or Command Prompt).Navigate to your project folder:cd path/to/your/OWN-FLAPPY/folder
Install all the required libraries using the requirements.txt file. Use the Python command for your specific version (e.g., py -3.10).
py -3.10 -m pip install -r requirements.txt

Step 2: Collect Training DataRun the Data Collector script:py -3.10 DataCollector.py
A webcam window will open. Click on this window to make it active.Press the following keys to collect data. Make sure to move your hand around (left, right, up, down) while collecting to get a good dataset.0 : Start collecting NEUTRAL gestures.1 : Start collecting THUMBS_UP gestures.2 : Start collecting THUMBS_DOWN gestures.s : Stop collecting the current gesture.q : Quit the script and save your data to gestures.csv.

Step 3: Train Your Deep Learning ModelRun the Model Trainer script. This will read your gestures.csv file, train the MLP model, and save all the model files (gesture_model.keras, scaler.pkl, etc.).py -3.10 ModelTrainer.py

Step 4: Play the Game!You're ready to play! Run the main script:py -3.10 main.py
The game and your webcam feed will open. Use your gestures to control the bird!(Optional)

 Step 5: Analyze Your ModelAfter training, you can run the analyzer to generate the plots for your paper (confusion_matrix.png, training_plots.png, etc.).py -3.10 ModelAnalyzer.py
