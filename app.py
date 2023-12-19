import gradio as gr
import nltk
import pandas as pd
nltk.download('punkt')
from fincat_utils import extract_context_words
from fincat_utils import bert_embedding_extract
import pickle
lr_clf = pickle.load(open("lr_clf_FiNCAT.pickle",'rb'))

def score_fincat(txt):
  '''
  Extracts numerals from financial texts and checks if they are in-claim or out-of claim

    Parameters:
      txt (str): Financial Text. This is to be given as input. Numerals present in this text will be evaluated.

    Returns:
      highlight (list): A list each element of which is a tuple. Each tuple has two elements i) word ii) whether the word is in-claim or out-of-claim.
      dff (pandas dataframe): A pandas dataframe having three columns 'numeral', 'prediction' (whether the word is in-claim or out-of-claim) and 'probability' (probabilty of the prediction).
  '''
  li = []
  highlight = []
  txt = " " + txt + " "
  k = ''
  for word in txt.split():
    if any(char.isdigit() for char in word):
      if word[-1] in ['.', ',', ';', ":", "-", "!", "?", ")", '"', "'"]:
        k = word[-1]
        word = word[:-1]
      st = txt.index(" " + word + k + " ")+1
      k = ''
      ed = st + len(word)
      x = {'paragraph' : txt, 'offset_start':st, 'offset_end':ed}
      context_text = extract_context_words(x)
      features = bert_embedding_extract(context_text, word)
      prediction = lr_clf.predict(features.reshape(1, 768))
      prediction_probability = '{:.4f}'.format(round(lr_clf.predict_proba(features.reshape(1, 768))[:,1][0], 4))
      highlight.append((word, '    In-claim' if prediction==1 else 'Out-of-Claim'))
      li.append([word,'    In-claim' if prediction==1 else 'Out-of-Claim', prediction_probability])
    else:
      highlight.append((word, '    '))
  headers = ['numeral', 'prediction', 'probability']
  dff = pd.DataFrame(li)
  dff.columns = headers
  return highlight, dff


iface = gr.Interface(fn=score_fincat, inputs=gr.inputs.Textbox(lines=5, placeholder="Enter Financial Text here..."), title="FiNCAT-2",description="Financial Numeral Claim Analysis Tool (Enhanced)", outputs=["highlight", "dataframe"], allow_flagging="never", examples=["In the year 2021, the markets were bullish. We expect to boost our sales by 80% this year.", "Last year our profit was $2.2M. This year it will increase to $3M"])
iface.launch()