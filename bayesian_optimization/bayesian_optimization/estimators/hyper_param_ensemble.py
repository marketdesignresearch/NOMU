import os.path
from collections import OrderedDict
from datetime import datetime

import numpy as np
from bayesian_optimization.utils.utils import timediff_d_h_m_s
from typing import *
from sklearn.model_selection import train_test_split
from bayesian_optimization.scores.scores import gaussian_nll_score, mse_score
from tensorflow.keras import backend as K



class HyperParamEnsemble:

    def __init__(
            self,
            context,
            base_model,
            test_size,
            kappa,
            K,
            epochs,
            batch_size,
            fixed_row_init,
            dropout_probability_range,
            l2reg_range,
            score,
            score_single_model,
            upper_bound_on_non_unique_models,
            global_seed: int = 1,
            name_prefix: str = "HDE_NN",
    ):
        self.base_model = base_model
        self.test_size = test_size
        self.kappa = kappa
        self.K = K
        self.epochs = epochs
        self.batch_size = batch_size
        self.upper_bound_on_non_unique_models = upper_bound_on_non_unique_models
        self.fixed_row_init = fixed_row_init
        self.dropout_probability_range = dropout_probability_range
        self.l2reg_range = l2reg_range
        self.name_prefix = name_prefix
        self.models = []
        self.model_dicts = []
        self.verbose = 1
        self.score = score
        self.score_single_model = score_single_model
        self.out_path = context.out_path
        self.global_seed = global_seed

    def initialize(self, x_train):
        first_row_models = []
        base_seed = self.global_seed
        for i in range(1, self.kappa + 1):
            if not self.fixed_row_init:
                base_seed = i
            modelname = f'{self.name_prefix}{base_seed}-{i}'
            dropout_prob = self._sample_dropout_probability()
            l2reg_factor = self._sample_l2reg_factor()
            copy_model = self.base_model.create_copy_advanced(
                seed_counter=base_seed,
                dropout_prob=dropout_prob,
                l2reg=self.get_l2reg_by_factor(l2reg_factor, dropout_prob, x_train),
                model_name=modelname
            )
            copy_model.set_context(self.base_model.context)
            first_row_models.append({
                "modelname": modelname,
                "dropout_probability": dropout_prob,
                "l2reg_factor": l2reg_factor,
                "model": copy_model,
                "base_seed": base_seed,
                "hyper_count": i
            })
            # print(f'{modelname}: dp:{dropout_prob}, l2:{l2reg_factor}, w1: {copy_model.model.get_weights()[0][0][0]}')
        return first_row_models

    def hyper_deep_ens(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            random_state: Optional[int] = None,
            verbose: int = 0
    ):
        K.clear_session()
        x_train, x_test, y_train, y_test = train_test_split(samples_x, samples_y, test_size=self.test_size,
                                                            random_state=random_state)
        start_ini = datetime.now()

        first_row_models = self.initialize(x_train)
        end_ini = datetime.now()
        # hyper_ens on first row
        self.fit_models(
            model_array=first_row_models,
            xtrain=x_train,
            ytrain=y_train)
        end_fi = datetime.now()
        self.predict_single_models(
            model_array=first_row_models,
            test_x=x_test)
        end_pred_1 = datetime.now()
        first_row_greedy = self.greedy(
            model_array=first_row_models,
            y=y_test
        )
        greedy_1 = datetime.now()
        stratified_rows_models = self.stratify(self.remove_duplicates(first_row_greedy), first_row_models, x_train)
        self.fit_models(
            model_array=stratified_rows_models,
            xtrain=x_train,
            ytrain=y_train)
        stratify = datetime.now()
        self.predict_single_models(
            model_array=stratified_rows_models,
            test_x=x_test)
        pred_rest = datetime.now()
        final_ensemble = self.greedy(
            model_array=first_row_models + stratified_rows_models,
            y=y_test
        )
        greedy_2 = datetime.now()
        if verbose > 0:
            os.makedirs(os.path.dirname(f'{self.out_path}/times.txt'), exist_ok=True)
            time_file = open(f'{self.out_path}/times.txt', 'a')
            time_file.write('Initialize: {}d {}h:{}m:{}s'.format(
                *timediff_d_h_m_s(end_ini - start_ini)) + f' len(first_row_models)={len(first_row_models)} \n')
            time_file.write('Fit First rRow: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(end_fi - end_ini)) + f' len(x_train)={len(x_train)} \n')
            time_file.write('Predict First Row: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(end_pred_1 - end_ini)) + '\n')
            time_file.write('Greedy 1: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(greedy_1 - end_pred_1)) + '\n')
            time_file.write('Stratify: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(stratify - greedy_1)) + '\n')
            time_file.write('Predict Rest: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(pred_rest - stratify)) + '\n')
            time_file.write('Greedy 2: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(greedy_2 - pred_rest)) + '\n')

        self.model_dicts = final_ensemble


    def fit_models(self, model_array: List, xtrain: np.array, ytrain: np.array):
        batch = self.batch_size
        if not batch:
            batch = len(xtrain)
        start = datetime.now()
        for model_dict in model_array:
            # print('Fit {}'.format(model_dict["modelname"]), batch)
            tmp = model_dict["model"].fit(x=xtrain, y=ytrain, epochs=self.epochs, verbose=0, batch_size=batch)
        end = datetime.now()
        diff = end - start
        print('Elapsed: {}d {}h:{}m:{}s'.format(*timediff_d_h_m_s(diff)),
              '(' + datetime.now().strftime("%H:%M %d-%m-%Y") + ')')

    @staticmethod
    def predict_single_models(model_array, test_x):
        for model_dict in model_array:
            pred_res = model_dict["model"].predict(test_x)
            if model_dict["model"].no_noise:
                model_dict["mu"] = pred_res
                model_dict["var"] = None
            else:
                model_dict["mu"] = pred_res[:, 0].reshape(-1, 1)
                model_dict["var"] = pred_res[:, 1].reshape(-1, 1)

    def greedy(self, model_array, y):
        score_current_ensemble = np.inf
        current_ensemble = []
        current_ensemble_size = 0
        # loop over unique values
        while len(self.remove_duplicates(current_ensemble)) < self.K and current_ensemble_size < self.upper_bound_on_non_unique_models:
            best_score = np.inf
            best_dict = None
            for model_dict in model_array:
                tmp_score = self.score_ensemble(current_ensemble + [model_dict], y)
                if tmp_score < best_score:
                    best_score = tmp_score
                    best_dict = model_dict
                if self.verbose > 1:
                    print(f'tmp_score:{tmp_score}')
                    print(f'best_score:{best_score}')
                    print(f'best_key:{best_dict["modelname"]}')
            key_best_nn = best_dict["modelname"]
            score_ensemble = best_score
            if self.verbose > 0: print(f'best:{key_best_nn} with', f'score of ensemble of:{score_ensemble}')
            # we want at least two distinct models in our ensemble -> the 'or'
            if score_ensemble <= score_current_ensemble or len(current_ensemble) == 1:
                if len(current_ensemble) == 1 and self.verbose > 0: print(
                    f'-> Forced adding of {key_best_nn}, since we want at least 2 NNs!')
                current_ensemble.append(best_dict)
                current_ensemble_size += 1
                score_current_ensemble = score_ensemble
            else:
                print('NO NN IMPROVES THE SCORE -> BREAK CONDITION')
                break
        # print break condition
        a = len(self.remove_duplicates(current_ensemble)) < self.K
        b = current_ensemble_size < self.upper_bound_on_non_unique_models
        if a and b:
            pass
        elif a and not b:
            print(f'BOUND of {self.upper_bound_on_non_unique_models} ON NON UNIQUE NNs REACHED -> BREAK CONDITION')
        elif not a and b:
            pp = self.K
            print(f'BOUND of {pp} ON UNIQUE NNs REACHED -> BREAK CONDITION')
        else:
            print('BOTH BOUNDS REACHED SIMULTANEOUSLY -> BREAK CONDITION')
        print(f'Return Ensemble:{[x["modelname"] for x in current_ensemble]}')
        return current_ensemble

    @staticmethod
    def remove_duplicates(ensemble):
        unique_ensemble = []
        distinct = []
        for model_dict in ensemble:
            if model_dict["modelname"] not in distinct:
                unique_ensemble.append(model_dict)
                distinct.append(model_dict["modelname"])
        return unique_ensemble

    def stratify(self, unique_ensemble, duplicate_ensemble, xtrain):
        stratified = []
        prev_seed = max([model_dict["base_seed"] for model_dict in duplicate_ensemble])
        for i in range(1, self.K + 1):
            for model_dict in unique_ensemble:
                hyper_c = model_dict["hyper_count"]
                modelname = f'{self.name_prefix}{i}-{hyper_c}'
                copy_model = self.base_model.create_copy_advanced(
                        seed_counter=prev_seed + i,
                        dropout_prob=model_dict["dropout_probability"],
                        l2reg=self.get_l2reg_by_factor(model_dict["l2reg_factor"], model_dict["dropout_probability"], xtrain),
                        model_name=modelname
                    )
                copy_model.set_context(self.base_model.context)
                stratified.append({
                    "modelname": modelname,
                    "dropout_probability": model_dict["dropout_probability"],
                    "l2reg_factor": model_dict["l2reg_factor"],
                    "model": copy_model,
                    "base_seed": prev_seed + i,
                    "hyper_count": hyper_c,
                })
                # print(f'{modelname}: dp:{model_dict["dropout_probability"]}, l2:{model_dict["l2reg_factor"]}, w1: {copy_model.model.get_weights()[0][0][0]}')
        return stratified

    def score_ensemble(self, ensemble, y):
        all_mu = [sub_dict["mu"] for sub_dict in ensemble]
        all_var = [sub_dict["var"] or 0 for sub_dict in ensemble]
        sum_mu_squared = sum(map(lambda x: x ** 2, all_mu))
        sum_mu = sum(all_mu)
        sum_var = sum(all_var)
        number_of_networks = len(ensemble)
        mu_pred = sum_mu / number_of_networks
        std_pred = np.sqrt((sum_var + sum_mu_squared) / number_of_networks - (sum_mu / number_of_networks) ** 2)
        # calc score for ensemble of size 1
        if len(ensemble) == 1:
            return mse_score(y, mu_pred)
        # calc score for ensemble of size > 1
        else:
            return gaussian_nll_score(y, mu_pred, std_pred)

    @staticmethod
    def get_l2reg_by_factor(factor, dropout_prob, xtrain):
        return ((1 - dropout_prob) / xtrain.shape[0]) * factor

    # def get_l2reg_by_index(self, index, first_row_models, xtrain):
    #     modelname = f'NN_1-{index}'
    #     dropout_prob = first_row_models[modelname]["dropout_probability"]
    #     l2reg_factor = first_row_models[modelname]["l2reg_factor"]
    #     return ((1 - dropout_prob) /xtrain.shape[0]) * l2reg_factor

    # def get_dropout_probability_index(self, index):
    #     modelname = f'NN_1-{index}'
    #     return self.first_row_models[modelname]["dropout_probability"]

    def _sample_dropout_probability(self):
        assert isinstance(self.dropout_probability_range, tuple), "dropout_probability_range must be a tuple"
        assert len(self.dropout_probability_range) == 2, "dropout_probability_range must have length 2"
        u = np.random.uniform(
            low=np.log10(self.dropout_probability_range[0]),
            high=np.log10(self.dropout_probability_range[1]),
            size=1
        )[0]  # 'log' uniform
        return 10 ** u

    def _sample_l2reg_factor(self):
        assert isinstance(self.l2reg_range, tuple), "l2reg_range must be a tuple"
        assert len(self.l2reg_range) == 2, "l2reg_range must have length 2"
        u = np.random.uniform(
            low=np.log10(self.l2reg_range[0]),
            high=np.log10(self.l2reg_range[1]),
            size=1
        )[0]  # 'log' uniform
        return 10 ** u
