from flask import Flask, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/')
def hello():
    return 'This is NLP Backend for FamilyDoctor Project'

@app.post('/')
def get_similar_label():
    print(request.get_json())
    data = request.get_json()
    answer = data['answer']
    labels = data['labels']
    return find_most_similar_label(answer, labels)



def find_most_similar_label(input_string, label_array, model_name='paraphrase-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)

    input_embedding = model.encode([input_string], convert_to_tensor = True)

    label_embeddings = model.encode(label_array, convert_to_tensor = True)

    similarities = util.pytorch_cos_sim(input_embedding, label_embeddings)[0]

    sorted_labels = [label for _, label in sorted(zip(similarities, label_array), reverse=True)]

    return sorted_labels[0]


if __name__ == '__main__':
    # answer = "مرحبا بك في معالجة اللغة الطبيعية"
    # labels = ["توليد النصوص", "استخراج المعرفة", "الترجمة الآلية", "تحليل النصوص"]
    # print(find_most_similar_label(answer, labels))
    app.run(port=9000, debug=True)