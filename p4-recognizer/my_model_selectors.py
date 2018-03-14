import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, n_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_n_components = self.n_constant
        return self.base_model(best_n_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    L is the likelihood of the fitted model.
    p is the number of parameters.
    N is the number of data points.

    BIC penalizes the model for complexity. The lower the BIC score the "better"
    the model. The term −2 log L decreases with increasing model complexity (more
    parameters), whereas the penalties p log N increase with increasing complexity.
    The BIC applies a larger penalty when N > e**2 = 7.4.
    """

    def calc_n_params(self, n_components, n_data_points):
        return ( n_components ** 2 ) + ( 2 * n_components * n_data_points ) - 1

    def calc_bic_score(self, logL, n_params, n_data_points):
        return (-2 * logL) + (n_params * np.log(n_data_points))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        best_bic_score = float("inf")
        n_data_points, n_features = self.X.shape
        log_n_data_points = np.log(n_data_points)

        # iterate through range of components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # create the model
            model = self.base_model(n_components)

            # score the model
            if model is not None:
                try:
                    # calculate the model score
                    logL = model.score(self.X, self.lengths)

                    # calculate BIC score
                    n_params = self.calc_n_params(n_components, n_data_points)
                    bic_score = self.calc_bic_score(logL, n_params, n_data_points)

                    # check BIC score against best model
                    if bic_score < best_bic_score:
                        best_bic_score, best_model = bic_score, model
                        if self.verbose:
                            print("word '{}': best model so far has {} states, BIC score = {}".format(
                                self.this_word, n_components, best_bic_score))
                except:
                    if self.verbose:
                        print("word '{}': model failed with {} states".format(
                            self.this_word, n_components))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf

    DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))

    "L" is likelihood of data fitting the model ("fitted" model)
    X is input training data given in the form of a word dictionary
    X(i) is the current word being evaluated
    M is a specific model

    Instead of a penalty term for complexity (as with BIC), DIC penalizes the model
    if liklihoods for non-matching words are too similar to model likelihoods for
    the correct word in the word set. The goal is to find the number of components
    where the difference is largest. The higher the DIC score the "better" the model.
    A high DIC scores means there is a high likelihood (small negative number)
    associated with the original word and a low likelihood (big negative number)
    with the other words in the dictionary.
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = None
        best_dic_score = float("-inf")

        # iterate through range of components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            # create the model
            model = self.base_model(n_components)

            # score the model
            if model is not None:
                try:
                    # calculate the model score
                    logL = model.score(self.X, self.lengths)

                    # calculate scores for all other words
                    word_count = 0
                    sum_logL = 0
                    for word, Xlength in self.hwords.items():
                        if word != self.this_word:
                            try:
                                sum_logL += model.score(Xlength)
                                word_count += 1
                            except:
                                if self.verbose:
                                    print("word '{}': model failed with {} states".format(
                                        word, n_components))

                    # calculate DIC score
                    if word_count > 1:
                        dic_score = logL - sum_logL / (word_count-1)
                    else:
                        dic_score = logL

                    # check DIC score against best model
                    if best_dic_score < dic_score:
                        best_dic_score, best_model = dic_score, model
                        if self.verbose:
                            print("word '{}': best model so far has {} sta  tes, DIC score = {}".format(
                                self.this_word, n_components, best_dic_score))
                except:
                    if self.verbose:
                        print("word '{}': model failed with {} states".format(
                            self.this_word, n_components))

        return best_model


class SelectorCV(ModelSelector):
    ''' Select best model based on average log Likelihood of cross-validation folds.
        The higher the CV score the "better" the model, although the model will
        likely overfit as complexity is added.

        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    '''

    def calc_best_cv_score(self, cv_score):
        # Max of list of lists comparing each item by value at index 0
        return max(cv_score, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        logLs = []
        cv_scores = []
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                # verify there's enough data
                if len(self.sequences) >= 3:
                    # break training set into folds
                    for train_idx, test_idx in kf.split(self.sequences):
                        # combine training sequences
                        self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                        # combine test sequences
                        X_test, lengths_test = combine_sequences(test_idx, self.sequences)

                        hmm_model = self.base_model(n_components)
                        logL = hmm_model.score(X_test, lengths_test)
                else:
                    hmm_model = self.base_model(n_components)
                    logL = hmm_model.score(self.X, self.lengths)

                logLs.append(logL)
                cv_score_avg = np.mean(logLs)
                cv_scores.append(tuple([cv_score_avg, hmm_model]))

            except Exception as e:
                pass
        return self.calc_best_cv_score(cv_scores)[1] if cv_scores else None
