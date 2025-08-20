from flask import Flask, jsonify, request, render_template
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

# Home page (frontend)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tcs", methods=["POST"])
def tcs_chatbot_api():
    data = request.get_json()
    question = data.get("tcs_question", "")

    # Load embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    new_vector_store = FAISS.load_local(
        "tcs_doc_index", embeddings, allow_dangerous_deserialization=True
    )

    # Search for relevant context
    context = new_vector_store.similarity_search(question, k=1)
    prompt = f"""
    Answer the question using only the given context from TCS Annual Reports 2024 and 2025.
    answer the question related to the pdf also answer the casul questions like ceo, the located tcs companies list , every year this company recruit how many members, how the the environment of tcs like answer all this casual questions  and dont answer when the question is related to other than any company and dont answer if you dont know that much about the question
    
    Context: {context}
    Question: {question}
    """

    try:
        # OpenAI API call
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for TCS Annual Reports."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
