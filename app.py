from flask import Flask, render_template, request, jsonify
from ask import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log', methods=['POST'])
def log():
    data = request.get_json()
    message = data['message']
    requete = main(message)

    print("✅ Sources:", requete["sources"])

    return jsonify(
        answer=requete["answer"],
        sources=requete["sources"]
    )

if __name__ == '__main__':
    app.run(debug=True)
