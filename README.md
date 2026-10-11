# College FAQ Chatbot

Created by Gaurav Kumar for the CodeAlpha AI internship.

## About
This chatbot answers common college questions using a sample FAQ collection.
All college information is fictional demo data.

## Features
- Chat interface with quick question buttons
- Searchable FAQs and topic filters
- Responses for unsupported or unclear questions
- Clear chat and download conversation

## Technologies
Python, Streamlit, NLTK and scikit-learn.

## How it works
NLTK processes the text. TF-IDF represents questions numerically,
and cosine similarity finds the closest matching FAQ.

## Installation
```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python -m streamlit run apps.py
```

Open the Local URL displayed in the terminal.

## Limitations
This chatbot matches stored FAQs and may misunderstand unfamiliar wording.
Replace sample answers with verified information before using it for a real college.