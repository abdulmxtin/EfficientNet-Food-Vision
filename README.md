🍕 Food Vision 101 – Image Classification
A deep learning project to classify images of food using the powerful EfficientNet architecture. This project showcases an end-to-end pipeline for training a state-of-the-art computer vision model on the Food-101 dataset.

📌 Features

EfficientNetB0 Model: Utilizes a pre-trained EfficientNet model, known for its high accuracy and computational efficiency.

Mixed-Precision Training: Leverages mixed-precision training on a Google Colab T4 GPU to significantly speed up the training process.

Food-101 Dataset: Trains on the comprehensive Food-101 dataset, which contains 101,000 images across 101 unique food categories.

Custom Helper Functions: Integrates custom utility functions from a GitHub repository for streamlined experiment tracking and visualization.

Inference: Includes a ready-to-use section to classify your own images after the model has been trained.

🛠️ Tech Stack

Python 🐍

TensorFlow 🧠

Numpy 🔢

Jupyter Notebook 📓

📂 Project Structure

Food_Vision_101/
│
├── Food_Vision_.ipynb # Main project notebook
├── helper_functions.py # Custom helper functions for the notebook
└── README.md
⚙️ How It Works

Data Preparation: The project directly downloads and extracts the Food-101 dataset from TensorFlow Datasets.

Model Training: A pre-trained EfficientNetB0 model is fine-tuned on the dataset, with callbacks for saving the best model and logging to TensorBoard.

Evaluation: The trained model's performance is evaluated to ensure it meets the required accuracy.

Inference: The final section of the notebook allows you to upload an image and receive a prediction from the trained model.

🚀 Getting Started

1️⃣ Open the notebook
Open the Food_Vision_.ipynb file in Google Colab.

2️⃣ Set up the environment
Navigate to Runtime > Change runtime type and select T4 GPU as the hardware accelerator.

3️⃣ Run all cells
Simply run each cell in the notebook from top to bottom. The code will handle everything from downloading the dataset to training the model and evaluating its performance.
