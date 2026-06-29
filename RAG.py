from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data["query"]

    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query)

    # Generate answer
    answer = generate_answer(query, retrieved_docs)

    return jsonify({
        "success": True,
        "query": query,
        "context": retrieved_docs,
        "answer": answer
    })

def retrieve_documents(query):
    return [
        "Relevant document 1",
        "Relevant document 2"
    ]

def generate_answer(query, docs):
    return f"Answer generated using retrieved documents for '{query}'."

if __name__ == "__main__":
    app.run(debug=True)
