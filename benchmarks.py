#!/usr/bin/env python3
"""
Precision at the indistinguishability threshold on two benchmark data sets.

Data (downloaded from OpenML on first run):
  Adult        (OpenML id 1590): income >50K, 24% positive.
  Mammography  (OpenML id 310):  microcalcifications, 2.3% positive.

For each data set, logistic regression and gradient boosting are trained on a
stratified half of the data and evaluated on the other half. All measures are
computed exactly from the test-set scores (no sampling):

  A        area under the ROC curve
  r_b      indistinguishability threshold, B(r_b) = 1/2
  C(r_b)   precision at r_b, with a 95% bootstrap interval
  v(r_b)   recall at r_b
  alpha    probability a true positive outscores a false positive at r_b
  PRBEP    precision (= recall) at the break-even point
  F1       F1 at r_b and the maximal F1

Finally, A is recomputed after removing the half of the test negatives with
the lowest scores (the 'easiest' negatives); C(r_b) is unchanged by this.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 2023
N_BOOT = 1000


def ranked_measures(s, y):
    """Exact threshold measures for scores s and labels y (1 positive, 0 negative).

    Thresholds are placed between distinct score values, so that the labelled
    set is always the k highest-scoring members, for k at the end of a tie group.
    """
    order = np.argsort(-s, kind='mergesort')
    s, y = s[order], y[order]
    P, n = y.sum(), len(y)
    pos_sorted = np.sort(s[y == 1])
    # For every member: proportion of the positive class scoring higher (ties count 1/2,
    # and a positive member is not compared with itself).
    above = P - np.searchsorted(pos_sorted, s, side='right')
    ties = np.searchsorted(pos_sorted, s, side='right') - np.searchsorted(pos_sorted, s, side='left')
    w = (above + 0.5 * (ties - y)) / P
    k = np.arange(1, n + 1)
    B = np.cumsum(w) / k                                  # B for the top-k labelled set
    TP = np.cumsum(y); FP = k - TP; FN = P - TP
    C = TP / k; v = TP / P
    F1 = 2 * TP / (2 * TP + FP + FN)
    ends = np.r_[s[1:] != s[:-1], True]                   # valid cut points (ends of tie groups)
    valid = np.where(ends)[0]
    # Smallest labelled set for which B >= 1/2 (B decreases as the threshold rises).
    kb = valid[np.argmax(B[valid] >= 0.5)]
    # alpha: P(true positive outscores false positive) within the labelled set
    negw = np.cumsum(np.where(y == 0, w * P, 0.0))        # sum over FPs of #positives above them
    alpha = negw[kb] / (TP[kb] * FP[kb]) if FP[kb] > 0 else np.nan
    kP = valid[np.argmin(np.abs(valid + 1 - P))]          # break-even: about P members labelled
    kF = valid[np.argmax(F1[valid])]
    A = w[y == 0].mean()                                  # for a negative, w = P(S_P > S_N)
    return dict(A=A, r_b=s[kb], k_b=kb + 1, P=P, C=C[kb], v=v[kb], alpha=alpha,
                FN_FP=FN[kb] / FP[kb] if FP[kb] > 0 else np.nan,
                F1_rb=F1[kb], PRBEP=C[kP], F1max=F1[kF], C_F1=C[kF], v_F1=v[kF])


def load(name):
    if name == 'Adult':
        d = fetch_openml(name='adult', version=2, as_frame=True)
        X = d.data
        y = (d.target == '>50K').astype(int).to_numpy()
    else:
        d = fetch_openml(data_id=310, as_frame=True)
        X = d.data.astype(float)
        y = (d.target.astype(str) == '1').astype(int).to_numpy()
    return X, y


def models(X):
    cat = X.select_dtypes(exclude='number').columns.tolist()
    num = X.select_dtypes(include='number').columns.tolist()
    pre = ColumnTransformer([('num', StandardScaler(), num),
                             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)])
    return {'Logistic regression': make_pipeline(pre, LogisticRegression(max_iter=2000)),
            'Gradient boosting': make_pipeline(pre, HistGradientBoostingClassifier(random_state=SEED))}


rng = np.random.default_rng(SEED)
rows, diffs = [], []
for name in ['Adult', 'Mammography']:
    X, y = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=SEED)
    scores = {}
    for mname, model in models(X).items():
        model.fit(Xtr, ytr)
        scores[mname] = model.predict_proba(Xte)[:, 1]
    # Paired bootstrap: both models are evaluated on the same resampled test sets.
    boot = {mname: {'C': [], 'A': []} for mname in scores}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yte), len(yte))
        for mname, s in scores.items():
            mb = ranked_measures(s[i], yte[i])
            boot[mname]['C'].append(mb['C']); boot[mname]['A'].append(mb['A'])
    for mname, s in scores.items():
        m = ranked_measures(s, yte)
        lo, hi = np.percentile(boot[mname]['C'], [2.5, 97.5])
        # Remove the easiest half of the negatives (lowest scores) and recompute.
        neg = np.where(yte == 0)[0]
        easiest = neg[np.argsort(s[neg])[:len(neg) // 2]]
        keep = np.setdiff1d(np.arange(len(yte)), easiest)
        m2 = ranked_measures(s[keep], yte[keep])
        rows.append((name, mname, m, lo, hi, m2))
    gb, lr = boot['Gradient boosting'], boot['Logistic regression']
    for key in ['C', 'A']:
        d = np.array(gb[key]) - np.array(lr[key])
        diffs.append((name, key, d.mean(), *np.percentile(d, [2.5, 97.5])))

print('%-12s %-20s %6s %6s | %6s %15s %6s %6s %7s | %6s %6s %6s | %6s %6s'
      % ('Data', 'Model', 'P', 'n_lab', 'C(rb)', '95% CI', 'v(rb)', 'alpha', 'FN/FP',
         'PRBEP', 'F1(rb)', 'maxF1', 'A', 'A_half'))
for name, mname, m, lo, hi, m2 in rows:
    print('%-12s %-20s %6d %6d | %6.3f  [%5.3f, %5.3f] %6.3f %6.3f %7.3f | %6.3f %6.3f %6.3f | %6.3f %6.3f'
          % (name, mname, m['P'], m['k_b'], m['C'], lo, hi, m['v'], m['alpha'], m['FN_FP'],
             m['PRBEP'], m['F1_rb'], m['F1max'], m['A'], m2['A']))
    assert abs(m2['C'] - m['C']) < 1e-12, 'C(r_b) should not change when easy negatives are removed'
    print('%35s check: FN/FP = %.4f, (2 alpha - 1) C = %.4f' % ('', m['FN_FP'], (2 * m['alpha'] - 1) * m['C']))

print('\nPaired bootstrap, gradient boosting minus logistic regression:')
for name, key, mean, lo, hi in diffs:
    print('  %-12s %-6s difference %+.3f  95%% CI [%+.3f, %+.3f]' % (name, 'C(r_b)' if key == 'C' else 'A', mean, lo, hi))
