import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer, util

# Load free local NLP model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Example FAQs
faqs = {
    "What are your opening hours?": "We are open from 9 AM to 6 PM, Monday to Saturday.",
    "Where are you located?": "We are located in Faisalabad, near D-Ground.",
    "Do you offer delivery services?": "Yes, we offer free home delivery within city limits.",
    "How can I contact support?": "You can contact us at support@example.com or call 123-4567890.",
    "What payment methods do you accept?": "We accept cash, cards, and online transfers.",
    "bye,exit":"Goodbye!See you soon",
}

faq_questions = list(faqs.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Function to get chatbot answer
def get_answer(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    best_match_idx = cosine_scores.argmax().item()
    best_question = faq_questions[best_match_idx]
    return faqs[best_question]

# Function to send message
def send_message():
    user_input = entry.get()
    if not user_input.strip():
        return
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You 🧑: {user_input}\n")
    entry.delete(0, tk.END)

    answer = get_answer(user_input)
    chat_window.insert(tk.END, f"Chatbot 🤖: {answer}\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

# Create GUI
root = tk.Tk()
root.title("FAQ Chatbot 🤖")
root.geometry("500x500")
root.configure(bg="#222831")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED,
                                        bg="#393E46", fg="white", font=("Segoe UI", 11))
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

frame = tk.Frame(root, bg="#222831")
frame.pack(fill=tk.X, padx=10, pady=5)

entry = tk.Entry(frame, bg="#EEEEEE", font=("Segoe UI", 11))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
entry.bind("<Return>", lambda event: send_message())

send_btn = tk.Button(frame, text="Send", command=send_message,
                     bg="#00ADB5", fg="white", font=("Segoe UI", 10, "bold"), width=10)
send_btn.pack(side=tk.RIGHT)

root.mainloop()
