# Next Word Predictor

🔮 **Next Word Predictor** is an LSTM-based language model web app built using PyTorch and Streamlit.  
It predicts the next word in a given sentence using a trained deep learning model.

---

## Features

- Predicts the next probable word given an input phrase.
- Lightweight model with vocabulary of 289 words.
- Simple and clean Streamlit user interface.
- Fast inference using PyTorch.
- Easily extendable for custom datasets and vocabularies.

---

## Demo

You can try the live demo on Hugging Face Spaces:  
[ https://huggingface.co/spaces/zeeshanbhutto89/next-word-predictor?logs=container ]

---

## Project Structure

Next_word_Api/
├── next-word-predictor/
│ ├── src/
│ │ ├── model.py # LSTM model definition
│ │ ├── vocab.pkl # Vocabulary dictionary file (word -> index)
│ │ ├── model.pth # Trained PyTorch model weights
│ │ ├── streamlit_app.py # Streamlit app code
│ │ └── pycache/ # Cache files (ignored in git)
│ ├── test_model.py # Script to test model and vocab loading
│ ├── test_vocab.py # Script to test vocab loading
│ ├── requirements.txt # Required Python packages
│ ├── Dockerfile # Docker configuration (optional)
│ └── README.md # This file


## Setup & Installation

1. Clone the repository:
   
   git clone https://github.com/yourusername/Next_word_Api.git
   cd Next_word_Api/next-word-predictor/src

2. Create and activate a virtual environment:

   python -m venv myenv
   source myenv/bin/activate      # On Windows: myenv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt


4. Run the Streamlit app locally:
   streamlit run streamlit_app.py


5. How to Test Model and Vocabulary
   Run the provided test scripts to ensure model and vocab files are correctly loaded:
   python test_vocab.py
   python test_model.py


6. Deployment

   Deployed on Hugging Face Spaces.

   Make sure your vocab.pkl and model.pth are properly uploaded and not corrupted.

   Update paths if you restructure directories.

7. Common Issues & Fixes


   UnpicklingError: Usually caused by corrupted vocab.pkl. Re-upload a clean file.


   RuntimeError while loading model: Indicates corrupted or incomplete model.pth. Re-save and upload the model.


   FileNotFoundError: Check your file paths and ensure files exist in correct directories.


   Streamlit Cache Problems: Use @st.cache_resource carefully; clear cache if needed.



Contributions
Feel free to submit issues and pull requests. For major changes, please open an issue first to discuss.

License
This project is licensed under the MIT License.

Contact
Created by Zeeshan Bhutto
[ https://www.linkedin.com/in/zeeshan-bhutto-475b43257/ ]
Email: zeeshanbhutto89@gmail.com

Thank you for checking out the Next Word Predictor! 🚀
