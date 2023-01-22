from flask import Flask, request, Response, jsonify, render_template
import pandas as pd
from collections import defaultdict
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
openai.api_key = os.getenv('OPENAI_KEY')
import numpy as np

app = Flask(__name__, template_folder="./frontend", static_folder="./frontend", static_url_path="")



def search_code(df, query, n=4):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.summary_embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    # df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    results = (
        df.sort_values("similarity", ascending=False)
    )
    return results.head(n)


def generate_answer(question):
  results = search_code(df, question, n=4)
  prompt = ''
  for i in range(3):
    prompt += results.iloc[i]["summary"] + "\n" + results.iloc[i]["blob"] + "\n"
  prompt += "\n" + "Answer the following question using the code context given above, and show an example with 'Example'\nQ: " + question + "\nA: "
  response = openai.Completion.create(
    model="text-davinci-003",
    # model="code-davinci-002",
    prompt=prompt,
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
  )
  return response["choices"][0]["text"]


def get_code_structure(df):
    kids_structure = defaultdict(list)
    parents_structure = {}
    for path in list(df['file_path'].unique()):
        t = path.split("/")
        for e in range(len(t)):
            if e < len(t) - 1:
              kids_structure[t[e]].append(t[e+1])
            else:
              kids_structure[t[e]] = "file"
            if e == 0:
              parents_structure[t[e]] = "./"
            else:
              parents_structure[t[e]] = "/".join(t[:e])

    for k, v in kids_structure.items():
        kids_structure[k] = list(set(v))
    return dict(kids_structure), parents_structure


# df = pd.read_csv("./frontend/data/embedded_summarized.csv")
df = pd.read_csv("./frontend/data/withsummary2.csv")
df['summary_embeddings'] = df['summary_embeddings'].apply(lambda x: eval(x))


df['embeddings'] = df['embeddings'].apply(lambda x: eval(x))

filetypes = ['py']


@app.route('/')
def home():
  stub = request.args.get('path', 'openpilot').strip()
  kids_structure, parents_structure = get_code_structure(df)
  if stub not in kids_structure:
    loctype = "nan"
    text = [["Path not available!"], [""]]
  elif any([stub.endswith(x) for x in filetypes]):
    loctype = "file"
    fullpath = f"{parents_structure[stub]}/{stub}"
    text = [[x, y] for x, y in zip(
      list(df[df['file_path'] == fullpath]['source']),
      list(df[df['file_path'] == fullpath]['summary'])
    )]
  else:
    loctype = "folder"
    text = [[x, ""] for x in kids_structure[stub]]
  res = {
    'parents': parents_structure[stub],
    'loctype': loctype,
    'text': text,
    'current': stub
  }
  return render_template('index.html', payload=res)



@app.route('/answer')
def answer():
  q = request.args.get('q', '').strip()
  a = search_code(df, q)
  res = [{'blob': x['blob'], 'summary': x['summary']} for x in a.to_dict('records')]

  return jsonify(res)

@app.route('/explain')
def explain():
  q = request.args.get('q', '').strip()
  a = generate_answer(q)
  return jsonify(a)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
