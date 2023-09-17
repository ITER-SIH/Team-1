from flask import Flask, render_template, request, jsonify, send_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__, template_folder="E:/vectorization", static_folder="static")

tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

@app.route("/")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.form.get("userMessage")
    if user_input.lower() == "exit":
        return "Goodbye!"
    
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    max_length = 500

    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = tokenizer.decode(output[0], skip_special_tokens=True)[:200]

    return jsonify({"response": response})

@app.route('/main')
def main_page():
    return send_file('main.html')

@app.route('/landing')
def landing_page():
    return send_file('landing.html')

@app.route('/login')
def login_page():
    return send_file('login.html')

@app.route('/signup')
def signup_page():
    return send_file('signup.html')

@app.route('/pricing')
def pricing_page():
    return send_file('pricing.html')

@app.route('/documentation')
def documentation_page():
    return send_file('documentation.html')

@app.route('/about')
def about_page():
    return send_file('about.html')

if __name__ == "__main__":
    app.run(debug=True)
