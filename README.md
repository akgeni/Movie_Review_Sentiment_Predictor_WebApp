![Language](https://img.shields.io/badge/language-Python-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md) 
# Movie_Review_Sentiment_Predictor_WebApp

A web application in Flask backed by LSTM, trained over acllmdb dataset predicts
whether sentiment in the review is positive or negative. 

LSTM of 128 hidden layers with recurrent dropout and sigmoid activation is trained to capture long term contexts.
It stores all the user reviews in sqlite database so that model can be improved later.

| Prediction | Actual | Text |
| --- | --- | --- |
| 1 | 1 | mission impossible 2 rocks ! ! ... . |
| 0 | 0 | she helped me bobbypin my insanely cool hat to my head , and she laughed at my stupid brokeback mountain cowboy jokes.. |
| 0 | 0 | oh , and brokeback mountain was a terrible movie . |





• Accuracy – 99.09% on test set.

It stores all the reviews in sqlite database so that model
can be improved later.

The project is live at : [http://akgeni.pythonanywhere.com/]
