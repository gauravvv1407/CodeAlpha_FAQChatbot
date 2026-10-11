import streamlit as st
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


faqs = [
    {
        "question": "What courses programs degrees are available?",
        "answer": "Our demo college offers BTech, BBA, BCA and MBA."
    },
    {
        "question": "How can I apply for admission?",
        "answer": "Fill out the application form and submit the required documents."
    },
    {
        "question": "What are the college opening hours timings?",
        "answer": "The college is open Monday to Saturday, from 9 AM to 5 PM."
    },
    {
        "question": "What are the tuition fees cost?",
        "answer": "Fees depend on the course. Contact the admissions office for details."
    },
    {
        "question": "Is hostel accommodation available?",
        "answer": "Yes, our demo college provides hostel accommodation."
    },
    {
        "question": "Does the college offer scholarships?",
        "answer": "Our demo college offers merit-based scholarships."
    },
    {
        "question": "What documents are required for admission?",
        "answer": "You need your marksheets, ID proof and passport-size photographs."
    },
    {
        "question": "Are placements jobs available?",
        "answer": "The placement cell helps students prepare for recruitment."
    },
    {
        "question": "Is there a library?",
        "answer": "Yes, students can access books and digital learning resources."
    },
    {
        "question": "Is bus transport available?",
        "answer": "Our demo college provides bus transport on selected routes."
    }
]



def clean_text(text):
    tokens = wordpunct_tokenize(text.lower())
    return " ".join(token for token in tokens if token.isalnum())


@st.cache_resource
def prepare_matcher():
    questions = [clean_text(faq["question"]) for faq in faqs]
    vectorizer = TfidfVectorizer(stop_words="english")
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors


vectorizer, question_vectors = prepare_matcher()


def get_answer(question):
    cleaned = clean_text(question)

    if cleaned in {"hi", "hello", "hey"}:
        return "Hello! Ask me about admissions, courses, fees or college facilities."

    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, question_vectors)[0]
    best_match = int(scores.argmax())

    if scores[best_match] < 0.3:
        return (
            "I couldn't find a suitable answer. "
            "Please rephrase or ask about college admissions and facilities."
        )

    return faqs[best_match]["answer"]


st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓")
st.title("🎓 College FAQ Chatbot")
st.caption("CodeAlpha AI Internship Project • Created by Gaurav Kumar")
st.info("Demo only: all college details are sample information.")

with st.sidebar:
    st.header("Try these questions")
    st.write("- Which courses are available?")
    st.write("- How do I apply for admission?")
    st.write("- What are the opening hours?")
    st.write("- Is there a hostel?")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a college question..."):
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.write(question)

    answer = get_answer(question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.write(answer)
        