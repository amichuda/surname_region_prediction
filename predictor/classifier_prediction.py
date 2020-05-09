import joblib
from pathlib import Path

import pandas as pd
from predictor.exceptions import NoTextException
import os

class ClassifierPredictor:
    
    def __init__(self, 
                 tfidf_path = None, 
                 model_path = None, 
                 label_encoder_path = None,
                 **kwargs):
        """This class uses a pickled trained classifier to make predictions about surnames.
        It loads the pickled processors (which include a tfidf transformer and label encoder),
        as well as the classifier and then trainsforms input text and give prediction

        Keyword Arguments:
            tfidf_path {str} -- path to trained tfidf transformer (joblib object) (default: {None})
            model_path {str} -- path to joblib pickle of trained model (default: {None})
            label_encoder_path {str} -- path to label encoder pickled object (default: {None})
        """
        
        super().__init__(**kwargs)
        
        if tfidf_path is None:
            tfidf_path = Path("predictor","saved_models", "tfidf_multilabel_False_nokampala_True.joblib")
        else:
            tfidf_path = Path(tfidf_path)
            
        if label_encoder_path is None:
            label_encoder_path = Path("predictor", 'saved_models', 'label_encoder_multilabel_False_nokampala_True.joblib')
        else:
            label_encoder_path = Path(label_encoder_path)
        
        if model_path is None:
            model_path = Path("predictor","saved_models", "xgb_None_multilabel_False_add_kampala_True.joblib")
        else:
            model_path = Path(model_path)
            
        self.tfidf_path = tfidf_path
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
            
    def load_tfidf(self):
        """Loads Tfidf transformer. See 
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        for information on all functionality.
        """
        
        return joblib.load(self.tfidf_path)
    
    def load_model(self):
        """Loads pickled trained classifier. Current one is an XGBoost classifier.
            See: https://xgboost.readthedocs.io/en/latest/ for more details.
        """
        return joblib.load(self.model_path)
    
    def load_label_encoder(self):
        """Loads label encoder. See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            for more information

        """
        return joblib.load(self.label_encoder_path)
    
    def process_text(self, text):
        """Pre-processes input text to make it ready for prediction.

        Arguments:
            text {str or list-like object} -- The input string or list of strings that are to be processed

        Returns:
            list -- Pre-processed list of strings
        """
        
        if isinstance(text, str):
            
            text = [text]
        
        processed_text = [i.lower().rstrip().lstrip() for i in text]
        
        return processed_text
    
    def transform_text(self, text = None):
        
        if text is None:
            raise NoTextException("No text was given for transformation.")
        
        tfidf = self.load_tfidf()
        
        processed_text = self.process_text(text)
        
        return tfidf.transform(processed_text)
        
    def predict(self, 
                text = None, 
                get_label_names = False,
                predict_prob = False,
                df_out = False):
        """Predicts origin based on classifier

        Keyword Arguments:
            text {list} -- List of strings to be predicted (default: {None})
            get_label_names {bool} -- Whether to output the label names after prediction (default: {False})
            predict_prob {bool} -- whether to give probabilities of coming from each region (default: {False})
            df_out {bool} -- whether to output a pandas dataframe (default: {False})
            
            >>> from predictor.classifier_prediction import ClassifierPredictor
            >>> # Instantiate predictor
            >>> c = ClassifierPredictor()
            >>> # Predict
            >>> prediction = c.predict(['Auma'])
            >>> print(prediction)

        Returns:
            pandas.DataFrame or list -- An object that contains predictions from the model
        """
        labels = self.load_label_encoder()
        
        X = self.transform_text(text)
        
        model = self.load_model()
        
        if get_label_names:
            if predict_prob:
                label_names = labels.classes_
                probs = model.predict_proba(X)
                result = {
                    t : {
                        label_name : prob for label_name, prob \
                        in zip(label_names, probs[i])
                        } for t, i in zip(text, range(len(probs)))
                    }
                
            else:
                result = labels.inverse_transform(model.predict(X))
        else:
            if predict_prob:
                result = model.predict_proba(X)
            else:
                result = model.predict(X)
        
        if df_out:
            
            return pd.DataFrame(result).T.sort_index()
        else:
            return result
        