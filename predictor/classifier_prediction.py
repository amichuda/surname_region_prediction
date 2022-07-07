from cProfile import label
import joblib
from pathlib import Path
from typing import Union, TypeVar

import pandas as pd
from predictor.exceptions import NoTextException
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

Vector = TypeVar('Vector')


class ClassifierPredictor:
    def __init__(
        self,
        column_name : str = None,
        tfidf_path : Union[str, Path] =None,
        model_path : Union[str, Path]=None,
        label_encoder_path : Union[str, Path] =None,
        agro_eco : bool=False,
        gaul : bool =False,
        calibrate : bool =True,
        **kwargs
    ):
        """A class that generates predictions based on a trained XGBoost model. Can predict based on Ugandan Regions, GAUL regions, or FAO agro-ecological zones. Optionally predicts gaul regions using an ensemble calibrated classifier.
        
        Example:
            >>> surnames = pd.DataFrame({'names':['Ahimbisibwe', 'Auma', 'Amin', 
                         'Makubuya', 'Museveni', 'Oculi', 'Kadaga']})
            >>> c = ClassifierPredictor(column_name='names')
            >>> predict_xgb = c.predict(surnames, 
                                        get_label_names=True, 
                                        predict_prob = True,
                                        df_out =True)

        Args:
            column_name (str, optional): When passing a pandas dataframe, the name of the columns with surnames. Defaults to None.
            tfidf_path (Union[str, Path], optional): the path of the joblib dump of the tfidf transformer. Defaults to None.
            model_path (Union[str, Path], optional): the path of the joblib dump of the trained model. Defaults to None.
            label_encoder_path (Union[str, Path], optional): the path of the joblib dump of the label encoder. Defaults to None.
            agro_eco (bool, optional): Whether to predict agro-ecological zones. Defaults to False.
            gaul (bool, optional): whether to predict gaul regions. Defaults to False.
            calibrate (bool, optional): whether to predict gaul regions. Defaults to True.
        """              
        if agro_eco ==True and gaul==True:
            
            raise Exception("You can't have agro_eco and gaul both true")
        

        if tfidf_path is None:
            
            if agro_eco:
                # tfidf_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "tfidf_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib",
                # )
                tfidf_path = "predictor/saved_models/tfidf_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib"
            elif gaul:
                # tfidf_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "tfidf_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib",
                # )
                
                tfidf_path = "predictor/saved_models/tfidf_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib"

            else:
                # tfidf_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "tfidf_multilabel_False_nokampala_True.joblib",
                # )
                
                tfidf_path = "predictor/saved_models/tfidf_multilabel_False_nokampala_True.joblib"
        else:
            tfidf_path = Path(tfidf_path)

        if label_encoder_path is None:

            if agro_eco:
                # label_encoder_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "label_encoder_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib",
                # )
                label_encoder_path = "predictor/saved_models/label_encoder_multilabel_False_nokampala_True_agro_zone_smote_False_opt.joblib"
            elif gaul:
                # label_encoder_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "label_encoder_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib",
                # )
                
                label_encoder_path = "predictor/saved_models/label_encoder_multilabel_False_nokampala_True_gaul_smote_False_gaul_opt.joblib"
                
            else:
                # label_encoder_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "label_encoder_multilabel_False_nokampala_True.joblib",
                # )
                
                label_encoder_path = "predictor/saved_models/label_encoder_multilabel_False_nokampala_True.joblib"
        else:
            label_encoder_path = Path(label_encoder_path)

        if model_path is None:

            if agro_eco:
                # model_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "xgb_None_multilabel_False_add_kampala_True_agro_zone_smote_False_opt.joblib",
                # )
                model_path = "predictor/saved_models/xgb_None_multilabel_False_add_kampala_True_agro_zone_smote_False_opt.joblib"
            elif gaul:
                if calibrate:
                    # model_path = Path(
                    #     "predictor",
                    #     'saved_models',
                    #     'xgb_None_calibrated_gaul_opt.joblib'
                    # )
                    model_path = "predictor/saved_models/xgb_None_calibrated_gaul_opt.joblib"
                else:
                    # model_path = Path(
                    #     "predictor",
                    #     "saved_models",
                    #     "xgb_None_multilabel_False_add_kampala_True_gaul_smote_False_gaul_opt.joblib"
                    # )
                    model_path = "predictor/saved_models/xgb_None_multilabel_False_add_kampala_True_gaul_smote_False_gaul_opt.joblib"
                    
            else:
                # model_path = Path(
                #     "predictor",
                #     "saved_models",
                #     "xgb_None_multilabel_False_add_kampala_True.joblib",
                # )
                model_path = "predictor/saved_models/xgb_None_multilabel_False_add_kampala_True.joblib"
        else:
            model_path = Path(model_path)

        self.tfidf_path = tfidf_path
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        self.column_name = column_name

    def load_tfidf(self) -> TfidfVectorizer:
        """Loads Tfidf transformer. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) for information on all functionality.
        
        Returns:
            TfidfVectorizer: A sklearn Tfidf Vectorizer Object
        """

        return joblib.load(self.tfidf_path)

    def load_model(self) -> Union[XGBClassifier, CalibratedClassifierCV]:
        """Loads pickled trained classifier. Current one is an XGBoost classifier.
            See [here](https://xgboost.readthedocs.io/en/latest/) for more details.
            
        Returns:
            Union[XGBClassifier, CalibratedClassifierCV]: Depending on the option, either a trained XGBoost Classifier or an sklearn calibrated classifier.
        """
        return joblib.load(self.model_path)

    def load_label_encoder(self) -> CountVectorizer:
        """Loads label encoder. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
            for more information.
            
        Returns:
            CountVectorizer: An sklearn Count Vectorizer object

        """
        return joblib.load(self.label_encoder_path)

    def process_text(self, text: Union[str, list]) -> list:
        """Pre-processes input text to make it ready for prediction.

        Arguments:
            text (str or list-like object): The input string or list of strings that are to be processed

        Returns:
            list: Pre-processed list of strings
        """

        if isinstance(text, str):

            text = [text]
        elif isinstance(text, pd.DataFrame):
            if self.column_name is None:
                raise Exception("Got a dataframe, but did not get `column_name`")
            text = text[self.column_name].tolist()

        processed_text = [i.lower().rstrip().lstrip() for i in text]

        return processed_text

    def transform_text(self, text : str = None) -> Vector:
        """A function that takes a list of surnames or strings and transforms them through the tf-idf transformer

        Args:
            text (str, optional): the text to be transformed. Defaults to None.

        Raises:
            NoTextException: Raises if no text was given

        Returns:
            Vector: A sparse matrix of numpy matrix
        """        

        if text is None:
            raise NoTextException("No text was given for transformation.")

        tfidf = self.load_tfidf()

        return tfidf.transform(text)

    def predict(self, 
                text : list =None, 
                get_label_names : bool=False, 
                predict_prob : bool=False, 
                df_out : bool=False) -> Union[pd.DataFrame, list]:        
        """Predicts origin based on classifier

        Keyword Arguments:
            text (list): List of strings to be predicted (default: {None})
            get_label_names (bool): Whether to output the label names after prediction (default: {False})
            predict_prob (bool): whether to give probabilities of coming from each region (default: {False})
            df_out (bool): whether to output a pandas dataframe (default: {False})
            
            >>> from predictor.classifier_prediction import ClassifierPredictor
            >>> # Instantiate predictor
            >>> c = ClassifierPredictor()
            >>> # Predict
            >>> prediction = c.predict(['Auma'])
            >>> print(prediction)

        Returns:
            Union[pd.DataFrame, list]: An object that contains predictions from the model
        """
        labels = self.load_label_encoder()
        raw_text_aux = text.copy(deep = True)
        text = self.process_text(text)
        raw_text = raw_text_aux.assign(processed_text = text).set_index('processed_text')
        X = self.transform_text(text)

        model = self.load_model()

        if get_label_names:
            if predict_prob:
                label_names = labels.classes_
                probs = model.predict_proba(X)
                result = {
                    t: {
                        label_name: prob
                        for label_name, prob in zip(label_names, probs[i])
                    }
                    for t, i in zip(text, range(len(probs)))
                }

            else:
                result = {
                    name: prediction
                    for name, prediction in zip(
                        text, labels.inverse_transform(model.predict(X))
                    )
                }
        else:
            if predict_prob:
                result = model.predict_proba(X)
            else:
                result = model.predict(X)

        if df_out:
            try:
                return (
                    pd.DataFrame(result).T
                    .sort_index()
                    .merge(raw_text, 
                           left_index = True, 
                           right_index = True)
                    .reset_index()
                    .set_index(self.column_name)
                    .rename({'index' : 'processed_surname'}, axis=1)
                    )
                
            except ValueError:
                return pd.DataFrame(result.values(), index = result.keys()).rename({0 : 'prediction'}, axis=1)
        else:
            return result

