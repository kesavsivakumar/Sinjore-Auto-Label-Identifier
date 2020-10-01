#!/usr/bin/python
# -*- coding: utf-8 -*-
import flask
from flask import Flask,render_template_string,request
from sklearn import preprocessing
import numpy as np
from tensorflow.keras.models import load_model
#from tensorflow.keras_retinanet.models import load_model
from scipy import misc
import tensorflow as tf
global graph,classifier
graph = tf.compat.v1.get_default_graph()
import re
import nltk
import string
#import string.maketrans
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
#from tensorflow.keras.optimizers import adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import csv 
import datetime
import pickle
from sklearn.externals import joblib
import os
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
        #index.html file is located in the  /template folder
	return flask.render_template(r'index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        text= request.form['ParagraphContent']
        if text==None: 
            return render_template('index.html', label="No  Text")
        with open(r'vector.pkl', 'rb') as f:
                cv= pickle.load(f)
        inp=pre_process(text)
        if(re.match("AJNS*",str(text))!=None):
                return render_template_string('A_ID - '+text)
        if(re.match("Author Queries",str(text))!=None or re.match("AQ[0-9][0-9]*:*",str(text))!=None):
                return render_template_string('AQ - '+text)
        if(re.match("abstract",str(text).lower())!=None):
                return render_template_string('ABSH - '+text)
        if(re.match("how to cite this article*",str(text).lower())!=None):
                return render_template_string('HTC - '+text)
        if(re.match("received:*",str(text).lower())!=None or re.match("accepted:*",str(text).lower())!=None or re.match("revised:*",str(text).lower())!=None):
                return render_template_string('HIS - '+text)
        if(re.match("figure [0-9]*[0-9]*:*",str(text).lower())!=None):
                return render_template_string('FGC - '+text)
        if(re.match("table [0-9]*[0-9]*:*",str(text).lower())!=None):
                return render_template_string('Normal - '+text)
        if(re.match("address for correspondence:*",str(text).lower())!=None):
                return render_template_string('ADD - '+text)   

        if(re.match("keywords*",str(text).lower())!=None):
                return render_template_string('KWD - '+text)
        

        

        X = cv.transform(inp).toarray()
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(r'Document_product_classes.npy')
        v1=OneHotEncoder(handle_unknown='ignore')
        v1.fit(np.asarray([[0],[1],[2],[3],[4]]))

        json_file = open(r'H1vsH2vsH3vsH4.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)

        json_file = open(r'10label.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model2 = model_from_json(model_json)

        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model = load_model(r'H1vsH2vsH3vsH4.h5')
        binary=model.predict(X)
        label=v1.inverse_transform(binary)
        tag=encoder.inverse_transform(label)
        if tag[0]=='Other':
                with open(r'vector_2.pkl', 'rb') as f:
                        cv2= pickle.load(f)
                X = cv2.transform(inp).toarray()
                encoder2 = preprocessing.LabelEncoder()
                encoder2.classes_ = np.load(r'Document_product_classes_2.npy')
                v2=OneHotEncoder(handle_unknown='ignore')
                v2.fit(np.asarray([[0],[1],[2],[3],[4],[5],[6],[7],[8]]))
                model2 = load_model(r'10label.h5')
                binary=model2.predict(X)
                label=v2.inverse_transform(binary)
                tag=encoder2.inverse_transform(label)       
        text= tag[0]+'  -  '+text
        return render_template_string(text)

def pre_process(text):
	corpus=[]
	text= re.split(r'\W+', str(text))
	text= re.sub('[0-9]',' ',str(text))
	text =re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]','',str(text))
	text = text.lower()
	text=word_tokenize(text)
	lemmatizer=WordNetLemmatizer()
	text=[lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
	text = ' '.join(text)
	corpus.append(text)
	return corpus	


if __name__=='__main__':
	#app.run(host='0.0.0.0',port=8000,debug=True)
	port = int(os.environ.get("PORT", 33507))
	app.run(host='0.0.0.0', port=port)
