from flask import Flask
from model.attentions import attention_api

# Create Flask app
app = Flask(__name__)

# Register blueprint
app.register_blueprint(attention_api)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
