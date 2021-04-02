from pathlib import Path
import pandas as pd
from fuzzywuzzy import process
import unicodedata

import numpy as np

from multiprocessing import Pool
from functools import partial
import itertools
from typing import Union

class TablePredictor:
    
    def __init__(self, 
                 agro_eco : bool = False, 
                 gaul : bool=False, 
                 table_path : Union[str, Path]  = None, 
                 column_name : str = None):
        """A class for predicting origin based on the frequency of name occurrences. This is implemented using a simple frequency tables based on the voter registration data.

        Args:
            agro_eco (bool, optional): Whether to predict agro-ecological zones. Defaults to False.
            gaul (bool, optional): whether to predict GAUL regions. Defaults to False.
            table_path (Union[str, Path], optional): Path to the CSV file with the frequency table. Defaults to None.
            column_name (str, optional): name of the column in the dataframe with the surname information. Defaults to None.

        """        
        
        if table_path is None:
            if agro_eco:
                table_path = Path('predictor',"table", "agro_zone_predictor.csv")
            elif agro_eco==False and gaul==True:
                table_path = Path('predictor', 'table', 'gaul_predictor.csv')
            elif agro_eco == True and gaul == True:
                raise Exception("Can't have agro_eco and gaul at the same time.")
            else:
                table_path = Path('predictor',"table", "table_predictor.csv")
            
            
        self.table_path = table_path
        
        self.table = self._load_table()
        
        self.column_name = column_name
        
    def _load_table(self) -> pd.DataFrame:
        """Loads table as attribute

        Returns:
            pd.DataFrame 
        """
        
        return pd.read_csv(self.table_path, index_col ='surname')
    
    def _process_text(self, text : Union[str, list]) -> dict:
        """Processes text for prediction

        Args:
            text (Union[str, list]): String to pre-process

        Returns:
            dict: Dictionary or mapping between input text and pre-processed text
        """
        
        if isinstance(text, str):
            text = [text]
            
        if isinstance(text, pd.DataFrame):
            if self.column_name is None:
                raise Exception("Got pandas dataframe, but did not specify `column_name`")
            
            text = text[self.column_name].tolist()
        
        processed_text = [t
                          .lower()
                          .lstrip()
                          .rstrip()
                          .replace(' ', '')
                          .replace('0', 'o')
                          .replace("'", '')
                          .replace(".", '')
                          for t in text]
        
        # Get rid of accents and diacritics
        processed_text = [unicodedata
                          .normalize('NFKD', t)
                          .encode('ascii', errors='ignore')
                          .decode('utf-8')
                          for t in processed_text]
        
        processed_text_dict = {processed_name : raw_name for raw_name, processed_name in zip(text, processed_text)}
        
        return processed_text_dict
    
    def _exact_match(self, text : list, data: pd.DataFrame) -> pd.DataFrame:
        """Tries to find exact match between input text and table

        Args:
            text (list): List of strings to find match
            data (pandas.DataFrame): pandas DataFrame with table of occurrence probabilities
            
            >>> from predictor.classifier_prediction import TablePredictor
            >>> # Instantiate predictor
            >>> t = TablePredictor()
            >>> # Predict
            >>> prediction = t.predict(['Auma'])
            >>> print(prediction)

        Returns:
            pandas.DataFrame: A pandas DataFrame of exact matches
        """
        
        result_df = (
            data[data.index.isin(text.keys())]
            .reset_index()
            )
        
        result_df['original_name'] = result_df.reset_index()['surname'].apply(lambda x: text.get(x))           

        result_df = result_df.set_index(['original_name']).rename({'surname' : 'processed_surname'}, axis= 1)
        
        return result_df
    
    def _fuzzy_match(self, data: pd.DataFrame, text: list) -> pd.DataFrame:
        """Finds fuzzy matched surnames and merges in table

        Arguments:
            data (pd.DataFrame): Input table of occurrence probabilities
            text (list): List of strings to fuzzy match

        Returns:
            pd.DataFrame: DataFrame with fuzzy matched occurrences
        """
        
        print("Fuzzy Matching...")
        # Find matches for all names
        matched = [process.extractOne(t, data.index) for t in text.keys()]
        
        extract_dict = {k:{'name_match':name, 'prob': prob} for k, (name, prob) in zip(text.values(), matched)}
        
        extract_df = pd.DataFrame(extract_dict).T
        return extract_df
    
    def _partition_table(self, data: pd.DataFrame, n: int) -> list:
        """Partitions data into pieces for multiprocessing

        Arguments:
            data (pandas.DataFrame): DataFrame to break up
            n (int): number of pieces to break into

        Returns:
            list -- List of DataFrames of input DataFrame broken into chunks
        """
        
        return np.array_split(data, n)    
        
              
    
    def predict(self, text: list, fuzzy : bool = False, n_jobs : int = 1) -> pd.DataFrame:
        """Predicts region probability by first trying to find an exact match and then 
        fuzzy matching for any names not matched.

        Arguments:
            text (list): List of strings to predict

        Keyword Arguments:
            n_jobs (int): The number of cores to use for fuzzy-matching. This
            only applies to fuzzy-matching. If all names are found by exact match, then
            multi-processing is not used at all. (default: {1})

        Returns:
            pd.DataFrame: DataFrame of predictions
        """
        
        processed_text = self._process_text(text = text)
        
        # First do exact match
        predicted_df = self._exact_match(text = processed_text, data = self.table)
        
        if fuzzy:
            # get dict of exact matches in same form as processed text
            predicted_dict = dict(map(reversed, predicted_df['processed_surname'].to_dict().items()))
            
            if predicted_df.index.size != len(processed_text):
                difference = len(processed_text) - predicted_df.index.unique().size
                print(f"Doing fuzzy matching for {difference} names...")
                
                # Get names that weren't found
                lost_dict = {k : processed_text[k] for k in set(processed_text) - set(predicted_dict)}
                
                if n_jobs == 1:
                    fuzzy_matches = self._fuzzy_match(text = lost_dict, data = self.table)
                    
                elif n_jobs > 1:
                    _fuzzy_lost_dict = partial(self._fuzzy_match, text = lost_dict)
                    
                    df_list = self._partition_table(data = self.table, n = n_jobs)
                    
                    pool = Pool(processes = n_jobs)
                    results = pool.map(_fuzzy_lost_dict, df_list)
                    
                    # Now get max probability for each name
                    results_df = pd.concat(results)
                    
                    max_df = (
                        results_df
                        .groupby(results_df.index)
                        .max()
                        .merge(self.table, left_on = 'name_match', right_index = True)
                    )
                    
                    predicted_df = predicted_df.append(max_df)
                    
                else:
                    raise Exception(f"{n_jobs} n_jobs not allowed as an input")
            
        return predicted_df
            
            
        
        
