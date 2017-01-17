![Language](https://img.shields.io/badge/language-Python-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md) 
# Movie_Review_Sentiment_Predictor_WebApp

A web application in Flask backed by ML models trained over acllmdb dataset predicts whether
sentiment in the review is positive or negative with % probability. 

SGDClassifier is trained using 
mini-batches of input data to scale better. 

It stores all the reviews in sqlite database so that model
can be improved later.

The project is live at : [http://akgeni.pythonanywhere.com/]
