import io
import os
import sys
import time
import logging
import argparse
import datetime
import sklearn
import numba
import numpy as np
import pandas as pd
import catboost as cb
from catboost import *


class utils:

    def __init__(self):
        pass


class CatBoost(utils):

    def __init__(self, booster_params=None, model_artifact=None, metric=None, objective=None, l2_leaf_reg=3,
                 bootstrap_type='Bayesian', bagging_temperature=1, subsample=0.66, sampling_frequency='PerTreeLevel',
                 random_strength=1, use_best_model=False, depth=6, ignored_features=None, one_hot_max_size=2,
                 has_time=False, rsm=1, nan_mode='Min', calc_feature_importance=True,
                 fold_permutation_block_size=1, leaf_estimation_iterations=None, leaf_estimation_method='Gradient',
                 fold_len_multiplier=2, approx_on_full_history=False, class_weights=None, thread_count=-1,
                 used_ram_limit=None, gpu_ram_part=0.95
                 **kwargs):
        '''
        Core parameters:

        :param booster_params: User-defined booster core parameters. This will override the default parameters.
        :param model_artifact: The path where model artifact was stored.
        :param metric: Supported metrics: RMSE, Logloss (default), MAE, CrossEntropy, Quantile, LogLinQuantile,
                       SMAPE, MultiClass — Not supported on GPU, MultiClassOneVsAll — Not supported on GPU, MAPE,
                       Poisson, PairLogit, QueryRMSE, QuerySoftMax.
        :param bootstrap_type: 'Poisson' (support GPU only), 'Bayesian', 'Bernoulli', 'No' type: string.
        :param bagging_temperature: Controls the intensity of Bayesian bagging. The higher the temperature, the more
               aggressive bagging will be. Type: float.
        :param subsample: Sample rate for bagging. This parameter can be used if one of the following bootstrap types
               is defined: 'Poisson', 'Bernoulli'. Type: float.
        :param subsample_frequency: Frequency to sample weights and objects when building trees. Supported values:
               'PerTree' and 'PerTreeLevel'. Type: string.
        :param random_strength: Score standard deviation multiplier. Type: float.
        :param use_best_model: If this parameter is set, the number of trees that are saved in the resulting model is
               defined as follows: 1. Build the number of trees defined by the training parameters. 2. Use the test data
               set to identify the iteration with the optimal value of the metric specified in 'eval_metric'. This
               option requires a test dataset. Type: bool.
        :param depth: Depth of the tree. The value can be any integer up to 16. It is recommended to use values in the
               range of 1 to 10. Type: int.
        :param ignored_features: Indices of features to exclude from training. The non-negative indices that do not
               match any features are successfully ignored. The identifier corresponds to the features index. Feature
               indices used in train and feature importance are numbered from 0 to feature count-1. Type: list.
        :param one_hot_max_size: Use one-hot-encoding for all features whose distinct values <= given hyper-parameter
               value. Type: int.
        :param has_time: Use the order of subjects in the input data (do not perform random permutation).The timestamp
               column type is used to determine the order of objects if specified. Type: bool.
        :param Random subspace method. The percentage of features to use at each split selection, when features are
               selected over again at random. Type: float.
        :param The method to process NaN within the input dataset. Support values: ('Forbidden': raise an exception,
               'Min', 'Max') Type: string.
        :param calc_feature_importance: This parameter turn on/off feature importance calculation. Type: bool.
        :param fold_permutation_block_size: Objects in the dataset are grouped in blocks before the random permutation.
               This parameter defines the size of the blocks. The smaller the value is, the slower the training will be.
               Too larger value may result in performance degradation. Type: int.
        :param leaf_estimation_iterations: The number of gradient steps when calculating the values in leaves. Type: int.
        :param leaf_estimation_method: The method used to calculate the values in leaves. Supported values: 'Newton',
               'Gradient'. Type: string.
        :param fold_len_multiplier: Coefficient for changing the length of folds. The value must be greater than 1.
               The best validation result is achieved with minimum values. With values close to 1, each iteration takes
               a quadratic amount of memory and time for the number of objects in the iteration.
               Thus, low values are possible only when there is a small number of objects.
        :param approx_on_full_history: The principles for calculating the approximated values. If set to True,
               will use all the preceding rows in the fold for calculating the approximated values.
               This mode is slower and in rare cases slightly more accurate. Type: bool.
        :param class_weights: Classes weights. The values are used as multipliers for the object weights.
               This parameter can be used for solving classification and multi-classification problems.
               For imbalanced datasets with binary classification the weight multiplier can be set to 1 for class 0 and
               (sum_negative/sum_positive) for class 1. For example,  class_weights=[0.1, 4] multiplies the weights of
               objects from class 0 by 0.1 and the weights of object from class 1 by 4.
        :param thread_count: The number of threads to use during training. The purpose depends on the selected processing unit:
               CPU: For CPU, Optimizes training time. This parameter doesn't affect results. For GPU, The given value is
               used for reading the data from the hard drive and does not affect the training. During the training one
               main thread and one thread for each GPU are used. GPU: The given value is used for reading the data from
               the hard drive and does not affect the training. During the training one main thread and one thread for
               each GPU are used. Type: int.
        :param used_ram_limit: The maximum amount of memory available to the CTR calculation process.
               Format: <size><measure of information>
               Supported measures of information (non case-sensitive): MB, KB, GB. Type: string.
        :param gpu_ram_part: How much of the GPU RAM to use for training. Type: float.

        :param objective:

        :param kwargs:
        '''
        self.__log_stream = io.StringIO()
        logging.basicConfig(stream=self.__log_stream, level=logging.INFO)

        self.__objective = objective
        self.__metric = metric

        if model_artifact != None:
            self.__model_artifact = model_artifact

        if booster_params != None:
            self.__booster_params = booster_params
        else:
            self.__booster_params = {
                'metric':self.__metric,
                'objective':self.__objective,
                'l2_leaf_reg':l2_leaf_reg,
                'bootstrap_type':bootstrap_type,
                'bagging_temperature':bagging_temperature,
                'subsample':subsample,
                'sampling_frequency':sampling_frequency,
                'random_strength':random_strength,
                'use_best_model':use_best_model,
                'depth':depth,
                'ignored_features':ignored_features,
                'one_hot_max_size':one_hot_max_size,
                'has_time':has_time,
                'rsm':rsm,
                'nan_mode':nan_mode,
                'calc_feature_importance':calc_feature_importance,
                'fold_permutation_block_size':fold_permutation_block_size,
                'leaf_estimation_iterations':leaf_estimation_iterations,
                'leaf_estimation_method':leaf_estimation_method,
                'fold_len_multiplier':fold_len_multiplier,
                'approx_on_full_history':approx_on_full_history,
                'class_weights':class_weights,
                'thread_count':thread_count
            }

    def get_log(self):
        """
        Retrive running log.
        :return:
        """
        __log = self.__log_stream.getvalue()
        return __log

    @classmethod
    def make_pool(cls, data, label=None, cat_vars=None, feature_names=None, weight=None, **kwargs):
        try:
            _pooled_data = Pool(data=data, label=label, cat_features=cat_vars, feature_names=feature_names,
                                weight=weight, **kwargs)
            return _pooled_data
        except Exception as e:
            logging.error('Failed in creating data pool.')
            raise

    def train(self, iterations=500, learning_rate=0.1, random_seed=8, early_stop=100, **kwargs):

        if early_stop != None:
            if isinstance(early_stop, int) == True:
                self.__booster_params['od_type'] = 'Iter'
                self.__booster_params['od_wait'] = early_stop
            else:
                logging.error('Provided an illegal early stop value.')
                raise TypeError('Early stop value can only be integers.')



