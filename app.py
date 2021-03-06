#!/usr/bin/python
# -*- coding: utf-8 -*-
import flask
from flask import Flask,render_template_string,render_template,request
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import numpy as np
from tensorflow.keras.models import load_model
#from tensorflow.keras_retinanet.models import load_model
from scipy import misc
import imageio
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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from skimage import color
from skimage import io
import csv 
import datetime
import pickle
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
        #index.html file is located in the  SASTRA_Covid/template folder
	return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        text= request.form['ParagraphContent']
        if text==None: 
            return render_template('index.html', label="No  Text")
        
        inp=pre_process(text)
        
        if(re.match("AJNS*",str(text))!=None):
                output='A_ID - '+text
                return render_template('index.html',output=output)
        if(re.match("Author Queries",str(text))!=None or re.match("AQ[0-9][0-9]*:*",str(text))!=None):
                output='AQ - '+text
                return render_template('index.html',output=output)
        if(re.match("abstract",str(text).lower())!=None):
                output='ABSH - '+text
                return render_template('index.html',output=output)
        if(re.match("how to cite this article*",str(text).lower())!=None):
                output='HTC - '+text
                return render_template('index.html',output=output)
        if(re.match("received:*",str(text).lower())!=None or re.match("accepted:*",str(text).lower())!=None or re.match("revised:*",str(text).lower())!=None):
                output='HIS - '+text
                return render_template('index.html',output=output)
        if(re.match("figure [0-9]*[0-9]*:*",str(text).lower())!=None):
                output='FGC - '+text
                return render_template('index.html',output=output)
        if(re.match("table [0-9]*[0-9]*:*",str(text).lower())!=None):
                output='Normal - '+text
                return render_template('index.html',output=output)
        if(re.match("address for correspondence:*",str(text).lower())!=None):
                output='ADD - '+text
                return render_template('index.html',output=output)
        if( re.match("introduction",str(text).lower())!=None or re.match("references",str(text).lower())!=None or re.match("discussion",str(text).lower())!=None or  re.match("conclusion",str(text).lower())!=None or re.match("case report.",str(text).lower())!=None or re.match("result.",str(text).lower())!=None or re.match("methods",str(text).lower())!=None or re.match("methodology",str(text).lower())!=None):
                output='H1 - '+text
                return render_template('index.html',output=output)
        if( re.match("conflicts of interest",str(text).lower())!=None or re.match("financial support and sponsorship",str(text).lower())!=None or re.match("statistical analysis",str(text).lower())!=None or re.match("acknowledgment",str(text).lower())!=None or re.match("declaration of patient consent",str(text).lower())!=None or re.match("case 1",str(text).lower())!=None or re.match("case 2",str(text).lower())!=None or re.match("limitation.",str(text).lower())!=None):
                output='H2 - '+text
                return render_template('index.html',output=output)
        if(re.match("[a-zA-Z0-9]*[,.]* et al.:",str(text).lower())!=None):
                output='RH - '+text
                return render_template('index.html',output=output)

        if(re.match("keywords*",str(text).lower())!=None):
                output='KWD - '+text
                return render_template('index.html',output=output)
        

        
        option = request.form['options']
        if option=='option1':
                output=H1_H2_H3(text)
        elif option=='option2':
                output=model2(text)
        elif option ==None:
                return render_template_string('the text could not be classified into any othe given fields please try click any of the models mentioned')

        return render_template('index.html',output=output)
              

     
def model2(text):
        with open(r'vector_2.pkl', 'rb') as f:
                cv= pickle.load(f)
        X = cv.transform([text]).toarray()
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(r'Document_product_classes_2.npy',allow_pickle=True)
        v1=OneHotEncoder(handle_unknown='ignore')
        v1.fit(np.asarray([[0],[1],[2],[3],[4],[5],[6]]))

        json_file = open(r'model2.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model = load_model(r'model2.h5')
        binary=model.predict(X)
        label=v1.inverse_transform(binary)
        tag=encoder.inverse_transform(label)
        text= tag[0]+'  -  '+text
        return text

       
def H1_H2_H3(text):
	
        with open(r'vector_H1-H2-H3.pkl', 'rb') as f:
                cv= pickle.load(f)
        X = cv.transform([text]).toarray()
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(r'Document_product_classes_H1-H2-H3.npy',allow_pickle=True)
        v1=OneHotEncoder(handle_unknown='ignore')
        v1.fit(np.asarray([[0],[1],[2]]))

        json_file = open(r'H1vsH2vsH3.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model = load_model(r'H1vsH2vsH3.h5')
        binary=model.predict(X)
        label=v1.inverse_transform(binary)
        tag=encoder.inverse_transform(label)
        text= tag[0]+'  -  '+text
        return text

def six_label(inp,text):
        with open(r'vector_6label.pkl', 'rb') as f:
                cv= pickle.load(f)
        X = cv.transform(inp).toarray()
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(r'Document_product_classes_6label.npy')
        v1=OneHotEncoder(handle_unknown='ignore')
        v1.fit(np.asarray([[0],[1],[2],[3],[4],[5]]))

        json_file = open(r'model-6label.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model = load_model(r'model-6label.h5')
        binary=model.predict(X)
        label=v1.inverse_transform(binary)
        tag=encoder.inverse_transform(label)
        text= tag[0]+'  -  '+text
        return text
        
def TX_ABS(inp,text):
        with open(r'vector_TX_vs_ABS.pkl', 'rb') as f:
                cv= pickle.load(f)
        X = cv.transform(inp).toarray()
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(r'Document_product_classes_TX_vs_ABS.npy')
        v1=OneHotEncoder(handle_unknown='ignore')
        v1.fit(np.asarray([[0],[1]]))

        json_file = open(r'model_TX_vs_ABS.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model = load_model(r'model_TX_vs_ABS.h5')
        binary=model.predict(X)
        label=v1.inverse_transform(binary)
        tag=encoder.inverse_transform(label)
        text= tag[0]+'  -  '+text
        return text      

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
	app.run(host='0.0.0.0',port=os.environ.get("PORT", 8000),debug=True)
