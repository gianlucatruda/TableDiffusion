import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pyemd import emd_samples
from scipy.stats import chisquare, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def _get_frequencies(real, synthetic):
    """Get percentual frequencies for each possible real categorical value.

    Based on https://github.com/sdv-dev/SDMetrics/blob/926bc8c276a7fc9dbc700d8f4a26947300343c7e/sdmetrics/utils.py#L40

    Given two iterators containing categorical data, this transforms it into
    observed/expected frequencies which can be used for statistical tests. It
    adds a regularization term to handle cases where the synthetic data contains
    values that don't exist in the real data.
    """
    f_obs, f_exp = [], []
    real, synthetic = defaultdict(float, Counter(real)), defaultdict(float, Counter(synthetic))

    # Calculate the sums outside the loop for efficiency.
    real_total, synthetic_total = sum(real.values()), sum(synthetic.values())

    for value in synthetic:
        real[value] += 1e-9  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / synthetic_total)
        f_exp.append(real[value] / real_total)

    return f_obs, f_exp


def compute_marginal_distances(real_dataset: pd.DataFrame, fake_dataset: pd.DataFrame, data_types):
    # TODO docstring
    scores = {}
    for _, (col, var_type) in enumerate(data_types):
        if "categorical" in var_type or "int" in var_type:
            # Inverted Chi-squared test (0.0 is best, 1.0 is worst)
            _fakes = fake_dataset[col]
            if "int" in var_type:
                _fakes.round(0)
            freq_obs, freq_exp = _get_frequencies(real_dataset[col], _fakes)
            if len(freq_obs) == len(freq_exp) == 1:
                pvalue = 1.0
            else:
                _, pvalue = chisquare(freq_obs, freq_exp)
            result = 1 - pvalue

        elif "float" in var_type:
            # Kolmogorov-Smirnov distance (0.0 is best, 1.0 is worst)
            statistic, _ = ks_2samp(real_dataset[col].fillna(0), fake_dataset[col].fillna(0))
            result = statistic
        else:
            warnings.warn(f"'{col}' is not a recognised type ({var_type}), ignoring...")

        scores[col] = result

    return pd.Series(scores)


def pmse_ratio(data, synthetic_data):
    """
    In order to determine how similar the synthetic and real data are
    to each other (general quality of synthetic) we can train a
    discriminator to attempt to distinguish between real and
    synthetic. The poorer the performance of the discriminator, the
    more similar the two datasets are.
    From "Really Useful Synthetic Data
    A Framework To Evaluate The Quality Of
    Differentially Private Synthetic Data"
    https://arxiv.org/pdf/2004.07740.pdf
    :param data: Original data
    :type data: pandas DataFrame
    :param synthetic_data: Synthetic data we are analyzing
    :type synthetic_data: pandas DataFrame
    :return: ratio (pmse score)
    :rtype: float
    """
    n1 = data.shape[0]
    n2 = synthetic_data.shape[0]
    comb = (
        pd.concat([data, synthetic_data], axis=0, keys=[0, 1])
        .reset_index(level=[0])
        .rename(columns={"level_0": "indicator"})
    )
    X_comb = comb.drop("indicator", axis=1)
    y_comb = comb["indicator"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_comb, y_comb, test_size=0.33, random_state=42
    )
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    score = clf.predict_proba(X_comb)[:, 1]
    observed_utility = sum((score - n2 / (n1 + n2)) ** 2) / (n1 + n2)
    expected_utility = clf.coef_.shape[1] * (n1 / (n1 + n2)) ** 2 * (n2 / (n1 + n2)) / (n1 + n2)
    return observed_utility / expected_utility


def sra(real, synth):
    """
    SRA can be thought of as the (empirical) probability of a
    comparison on the synthetic data being ”correct” (i.e. the same as
    the comparison would be on the real data).
    From "Measuring the quality of Synthetic data for use in competitions"
    https://arxiv.org/pdf/1806.11345.pdf
    (NOTE: SRA requires at least 2 accuracies per list to work)
    :param real: list of accuracies on models of real data
    :type real: list of floats
    :param synth: list of accuracies on models of synthetic data
    :type synth: list of floats
    :return: sra score
    :rtype: float
    """
    k = len(real)
    sum_I = 0
    for i in range(k):
        R_vals = np.array([real[i] - rj if i != k else None for k, rj in enumerate(real)])
        S_vals = np.array([synth[i] - sj if i != k else None for k, sj in enumerate(synth)])
        I = R_vals[R_vals != np.array(None)] * S_vals[S_vals != np.array(None)]
        I[I >= 0] = 1
        I[I < 0] = 0
        sum_I += I
    return np.sum((1 / (k * (k - 1))) * sum_I)


def wasserstein_randomization(d1_large, d2_large, iters=10, downsample_size=100):
    """
    Combine synthetic and real data into two sets and randomly
    divide the data into two new random sets. Check the wasserstein
    distance (earth movers distance) between these two new muddled sets.
    Use the measured wasserstein distance to compute the ratio between
    it and the median of the null distribution (earth movers distance on
    original set). A ratio of 0 would indicate that the two marginal
    distributions are identical.
    From "REALLY USEFUL SYNTHETIC DATA
    A FRAMEWORK TO EVALUATE THE QUALITY OF
    DIFFERENTIALLY PRIVATE SYNTHETIC DATA"
    https://arxiv.org/pdf/2004.07740.pdf
    NOTE: We return the mean here. However, its best
    probably to analyze the distribution of the wasserstein score
    :param d1_large: real data
    :type d1_large: pandas DataFrame
    :param d2_large: fake data
    :type d2_large: pandas DataFrame
    :param iters: how many iterations to run the randomization
    :type iters: int
    :param downsample_size: we downsample the original datasets due
    to memory constraints
    :type downsample_size: int
    :return: wasserstein randomization mean
    :rtype: float
    """
    # pip install pyemd
    # https://github.com/wmayner/pyemd

    assert len(d1_large) == len(d2_large)
    d1 = d1_large.sample(n=downsample_size)
    d2 = d2_large.sample(n=downsample_size)
    l_1 = len(d1)
    d3 = np.concatenate((d1, d2))
    distances = []
    for _ in range(iters):
        np.random.shuffle(d3)
        n_1, n_2 = d3[:l_1], d3[l_1:]
        # TODO: Readdress the constraints of PyEMD
        # For now, decrease bin size drastically for
        # more efficient computation
        # try:
        #     # PyEMD is sometimes memory intensive
        #     # Let's reduce bins if so
        #     dist = emd_samples(n_1, n_2, bins='auto')
        # except MemoryError:
        dist = emd_samples(n_1, n_2, bins=10)
        distances.append(dist)

    # Safety check, to see if there are any valid
    # measurements
    return np.mean(np.array(distances)) if distances else -1


def alpha_beta_auth(df_X, df_X_syn, emb_center=None):
    """
    Evaluates the alpha-precision, beta-recall, and authenticity scores.

    The class evaluates the synthetic data using a tuple of three metrics:
    alpha-precision, beta-recall, and authenticity.
    Note that these metrics can be evaluated for each synthetic data point (which are useful for auditing and
    post-processing). Here we average the scores to reflect the overall quality of the data.
    The formal definitions can be found in the reference below:

    Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. "How faithful is your synthetic
    data? sample-level metrics for evaluating and auditing generative models."
    In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
    """

    X = df_X.values
    X_syn = df_X_syn.values
    assert len(X) == len(X_syn)

    emb_center = np.mean(X, axis=0)

    n_steps = 30
    alphas = np.linspace(0, 1, n_steps)

    Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

    synth_center = np.mean(X_syn, axis=0)

    alpha_precision_curve = []
    beta_coverage_curve = []

    synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

    nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
    real_to_real, _ = nbrs_real.kneighbors(X)

    nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

    # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real = real_to_real[:, 1].squeeze()
    real_to_synth = real_to_synth.squeeze()
    real_to_synth_args = real_to_synth_args.squeeze()

    real_synth_closest = X_syn[real_to_synth_args]

    real_synth_closest_d = np.sqrt(np.sum((real_synth_closest - synth_center) ** 2, axis=1))
    closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

    for k in range(len(Radii)):
        precision_audit_mask = synth_to_center <= Radii[k]
        alpha_precision = np.mean(precision_audit_mask)

        beta_coverage = np.mean(
            ((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k]))
        )

        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)

    # See which one is bigger

    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen)

    Delta_precision_alpha = 1 - 2 * np.sum(
        np.abs(np.array(alphas) - np.array(alpha_precision_curve))
    ) * (alphas[1] - alphas[0])
    Delta_coverage_beta = 1 - 2 * np.sum(
        np.abs(np.array(alphas) - np.array(beta_coverage_curve))
    ) * (alphas[1] - alphas[0])

    return (
        Delta_precision_alpha,
        Delta_coverage_beta,
        authenticity,
    )
