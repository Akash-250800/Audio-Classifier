Audio Genre Classifier
A full pipeline Music Genre Classifier built using deep learning, audio processing, and Streamlit.
This project classifies .wav audio files into one of 10 music genres from the GTZAN Dataset.

🚀 Features
🔥 Used both Raw Audio (.wav) + CSV Features (features_30_sec.csv)

🎶 Extracted MFCC, Chroma, and Spectral Contrast features with Librosa

🔥 Upgraded to Mel-Spectrogram + CNN model (visual audio classification)

📈 Increased model accuracy from 22% ➔ 76.5%

📊 Plotted Confusion Matrix and Classification Report

🌐 Built an interactive Streamlit app for live audio genre prediction

⚙️ Setup GitHub Actions CI/CD workflow

✅ Properly used Git LFS to manage large model files

📚 Technologies Used

Technology	Purpose
Python 3.10	Core language
PyTorch	CNN Model training
Librosa	Audio feature extraction
Scikit-learn	Data processing, evaluation
Matplotlib	Visualizations (spectrograms, confusion matrix)
Streamlit	Web app interface
GitHub Actions	CI/CD Automation
Git LFS	Handling .pth model files
🎯 How It Works
Raw Audio: Extracted Mel Spectrograms from .wav files using Librosa.

CSV Features: Also experimented with traditional features from features_30_sec.csv.

Model Building:

Started with basic MLP (CSV features)

Upgraded to Convolutional Neural Network (CNN) (Mel Spectrograms)

Training: Trained the CNN to classify 10 music genres with ~76.5% accuracy.

Deployment: Built a Streamlit app for live audio file uploads and genre prediction.

🏗️ Project Structure
bash
Copy
Edit
Audio-Classifier/
├── app.py                 # Streamlit app
├── cnn_model.py           # CNN model architecture
├── utils.py               # Helper functions (feature extraction, loading)
├── model/
│    └── spectrogram_cnn.pth   # Trained model (Git LFS tracked)
├── requirements.txt       # Python dependencies
├── .github/
│    └── workflows/
│         └── deploy.yml    # GitHub Actions CI/CD workflow
└── README.md              # Project documentation
📈 Model Performance

Metric	Result
Initial Accuracy (CSV model)	~22%
Improved Accuracy (raw audio model)	~76.5%
Validation Method	Train/test split (80/20)
🌐 Streamlit App
You can upload a .wav file and the app will predict the music genre!

Coming soon: Launch App 🚀 ← (Add your Streamlit Cloud link here after deployment)

🛠️ Setup Instructions
Clone the repo:

bash
Copy
Edit
git clone https://github.com/Akash-250800/Audio-Classifier.git
cd Audio-Classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app locally:

bash
Copy
Edit
streamlit run app.py
✅ Make sure you have a model/ folder with the spectrogram_cnn.pth file inside.

🧠 Lessons Learned
How to process audio files into usable deep learning inputs

How to build and train CNN models from scratch

How to handle large models using Git LFS

How to automate deployments using GitHub Actions

How to create beautiful Streamlit web apps for ML projects

📜 License
This project is licensed under the MIT License.

🎉 Thank You!
Developed with ❤️ by Akash-250800
Powered by Python, Librosa, PyTorch, and Streamlit.

