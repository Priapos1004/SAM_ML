from typing import Union

import pandas as pd
from catboost import CatBoostClassifier

from .main_classifier import Classifier


class CBC(Classifier):
    def __init__(
        self,
        model_name: str = "CatBoostClassifier",
        iterations=None,
        learning_rate=None,
        depth=None,
        l2_leaf_reg=None,
        model_size_reg=None,
        rsm=None,
        loss_function=None,
        border_count=None,
        feature_border_type=None,
        per_float_feature_quantization=None,
        input_borders=None,
        output_borders=None,
        fold_permutation_block=None,
        od_pval=None,
        od_wait=None,
        od_type=None,
        nan_mode=None,
        counter_calc_method=None,
        leaf_estimation_iterations=None,
        leaf_estimation_method=None,
        thread_count=None,
        random_seed=None,
        use_best_model=None,
        verbose=0,
        logging_level=None,
        metric_period=None,
        ctr_leaf_count_limit=None,
        store_all_simple_ctr=None,
        max_ctr_complexity=None,
        has_time=None,
        allow_const_label=None,
        classes_count=None,
        class_weights=None,
        one_hot_max_size=None,
        random_strength=None,
        name=None,
        ignored_features=None,
        train_dir=None,
        custom_loss=None,
        custom_metric=None,
        eval_metric=None,
        bagging_temperature=None,
        save_snapshot=None,
        snapshot_file=None,
        snapshot_interval=None,
        fold_len_multiplier=None,
        used_ram_limit=None,
        gpu_ram_part=None,
        allow_writing_files=None,
        final_ctr_computation_mode=None,
        approx_on_full_history=None,
        boosting_type=None,
        simple_ctr=None,
        combinations_ctr=None,
        per_feature_ctr=None,
        task_type=None,
        device_config=None,
        devices=None,
        bootstrap_type=None,
        subsample=None,
        sampling_unit=None,
        dev_score_calc_obj_block_size=None,
        max_depth=None,
        n_estimators=None,
        num_boost_round=None,
        num_trees=None,
        colsample_bylevel=None,
        random_state=42,
        reg_lambda=None,
        objective=None,
        eta=None,
        max_bin=None,
        scale_pos_weight=None,
        gpu_cat_features_storage=None,
        data_partition=None,
        metadata=None,
        early_stopping_rounds=None,
        cat_features=None,
        grow_policy=None,
        min_data_in_leaf=None,
        min_child_samples=None,
        max_leaves=None,
        num_leaves=None,
        score_function=None,
        leaf_estimation_backtracking=None,
        ctr_history_unit=None,
        monotone_constraints=None,
        feature_weights=None,
        penalties_coefficient=None,
        first_feature_use_penalties=None,
        model_shrink_rate=None,
        model_shrink_mode=None,
        langevin=None,
        diffusion_temperature=None,
        posterior_sampling=None,
        boost_from_average=None,
        text_features=None,
        tokenizers=None,
        dictionaries=None,
        feature_calcers=None,
        text_processing=None,
    ):
        """
        @info:
            no type hints due to documentation of CatBoost library
        @param (important one):
            depth - depth of the tree
            learning_rate - learning rate
            iterations - maximum number of trees that can be built when solving machine learning problems
            bagging_temperature - defines the settings of the Bayesian bootstrap
            random_strength - the amount of randomness to use for scoring splits when the tree structure is selected
            l2_leaf_reg - coefficient at the L2 regularization term of the cost function
            border_count - the number of splits for numerical features
        """
        self.model_name = model_name
        self.model_type = "CBC"
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            model_size_reg=model_size_reg,
            rsm=rsm,
            loss_function=loss_function,
            border_count=border_count,
            feature_border_type=feature_border_type,
            per_float_feature_quantization=per_float_feature_quantization,
            input_borders=input_borders,
            output_borders=output_borders,
            fold_permutation_block=fold_permutation_block,
            od_pval=od_pval,
            od_wait=od_wait,
            od_type=od_type,
            nan_mode=nan_mode,
            counter_calc_method=counter_calc_method,
            leaf_estimation_iterations=leaf_estimation_iterations,
            leaf_estimation_method=leaf_estimation_method,
            thread_count=thread_count,
            random_seed=random_seed,
            use_best_model=use_best_model,
            verbose=verbose,
            logging_level=logging_level,
            metric_period=metric_period,
            ctr_leaf_count_limit=ctr_leaf_count_limit,
            store_all_simple_ctr=store_all_simple_ctr,
            max_ctr_complexity=max_ctr_complexity,
            has_time=has_time,
            allow_const_label=allow_const_label,
            classes_count=classes_count,
            class_weights=class_weights,
            one_hot_max_size=one_hot_max_size,
            random_strength=random_strength,
            name=name,
            ignored_features=ignored_features,
            train_dir=train_dir,
            custom_loss=custom_loss,
            custom_metric=custom_metric,
            eval_metric=eval_metric,
            bagging_temperature=bagging_temperature,
            save_snapshot=save_snapshot,
            snapshot_file=snapshot_file,
            snapshot_interval=snapshot_interval,
            fold_len_multiplier=fold_len_multiplier,
            used_ram_limit=used_ram_limit,
            gpu_ram_part=gpu_ram_part,
            allow_writing_files=allow_writing_files,
            final_ctr_computation_mode=final_ctr_computation_mode,
            approx_on_full_history=approx_on_full_history,
            boosting_type=boosting_type,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            task_type=task_type,
            device_config=device_config,
            devices=devices,
            bootstrap_type=bootstrap_type,
            subsample=subsample,
            sampling_unit=sampling_unit,
            dev_score_calc_obj_block_size=dev_score_calc_obj_block_size,
            max_depth=max_depth,
            n_estimators=n_estimators,
            num_boost_round=num_boost_round,
            num_trees=num_trees,
            colsample_bylevel=colsample_bylevel,
            random_state=random_state,
            reg_lambda=reg_lambda,
            objective=objective,
            eta=eta,
            max_bin=max_bin,
            scale_pos_weight=scale_pos_weight,
            gpu_cat_features_storage=gpu_cat_features_storage,
            data_partition=data_partition,
            metadata=metadata,
            early_stopping_rounds=early_stopping_rounds,
            cat_features=cat_features,
            grow_policy=grow_policy,
            min_data_in_leaf=min_data_in_leaf,
            min_child_samples=min_child_samples,
            max_leaves=max_leaves,
            num_leaves=num_leaves,
            score_function=score_function,
            leaf_estimation_backtracking=leaf_estimation_backtracking,
            ctr_history_unit=ctr_history_unit,
            monotone_constraints=monotone_constraints,
            feature_weights=feature_weights,
            penalties_coefficient=penalties_coefficient,
            first_feature_use_penalties=first_feature_use_penalties,
            model_shrink_rate=model_shrink_rate,
            model_shrink_mode=model_shrink_mode,
            langevin=langevin,
            diffusion_temperature=diffusion_temperature,
            posterior_sampling=posterior_sampling,
            boost_from_average=boost_from_average,
            text_features=text_features,
            tokenizers=tokenizers,
            dictionaries=dictionaries,
            feature_calcers=feature_calcers,
            text_processing=text_processing,
        )

    def hyperparameter_tuning(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        depth: list[int] = [4, 5, 6, 7, 8, 9, 10],
        learning_rate: list[float] = [0.1, 0.01, 0.02, 0.03, 0.04],
        iterations: list[int] = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            200,
            500,
            1000,
        ],
        bagging_temperature: list[float] = [0.1,0.2,0.3,0.4,0.6,0.8,1.0],
        random_strength: list[float] = [
            0.000000001,
            0.0000001,
            0.00001,
            0.001,
            0.1,
            1,
            10,
        ],
        l2_leaf_reg: list[int] = [2, 4, 6, 8, 12, 16, 20, 24, 30],
        border_count: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 254],
        n_split_num: int = 10,
        n_repeats_num: int = 3,
        verbose: int = 0,
        scoring: str = "accuracy",
        avg: str = "macro",
        pos_label: Union[int, str] = 1,
        rand_search: bool = True,
        n_iter_num: int = 75,
        console_out: bool = False,
        train_afterwards: bool = True,
    ):
        """
        @param:
            x_train - DataFrame with train features
            y_train - Series with labels

            depth - depth of the tree
            learning_rate - learning rate
            iterations - maximum number of trees that can be built when solving machine learning problems
            bagging_temperature - defines the settings of the Bayesian bootstrap
            random_strength - the amount of randomness to use for scoring splits when the tree structure is selected
            l2_leaf_reg - coefficient at the L2 regularization term of the cost function
            border_count - the number of splits for numerical features

            n_split_num - number of different splits
            n_repeats_num - number of repetition of one split

            scoring - metrics to evaluate the models
            avg - average to use for precision and recall score (e.g.: "micro", "weighted", "binary")
            pos_label - if avg="binary", pos_label says which class to score. Else pos_label is ignored
            rand_search - True: RandomizedSearchCV, False: GridSearchCV
            n_iter_num - Combinations to try out if rand_search=True

            verbose - log level (higher number --> more logs)
            console_out - output the the results of the different iterations
            train_afterwards - train the best model after finding it

        @return:
            set self.model = best model from search
        """
        # Create the random grid
        grid = dict(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
        )

        self.gridsearch(
            x_train=x_train,
            y_train=y_train,
            grid=grid,
            scoring=scoring,
            avg=avg,
            pos_label=pos_label,
            rand_search=rand_search,
            n_iter_num=n_iter_num,
            n_split_num=n_split_num,
            n_repeats_num=n_repeats_num,
            verbose=verbose,
            console_out=console_out,
            train_afterwards=train_afterwards,
        )
