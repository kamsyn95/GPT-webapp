from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, render_template


app = Flask(__name__)

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    # Get user input from form
    user_input = request.form['user_input']

    # Generate text from Language Model
    inputs = tokenizer.encode(user_input.strip(), return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=1.0)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return template with generated text
    return render_template('index.html', variable=output_text)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
