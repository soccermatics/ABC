# Precision at the indistinguishability threshold

This repository contains the code and article for *Precision at the indistinguishability threshold: a method for evaluating classification algorithms* by David J. T. Sumpter ([arXiv:2311.11422](https://arxiv.org/abs/2311.11422)).

There exist a wide range of single number metrics for assessing performance of classification algorithms, including AUC and the F1-score (Wikipedia lists 17 such metrics, with 27 different names). In this article, I propose a new metric to answer the following question: when an algorithm is tuned so that it can no longer distinguish labelled cats from real cats, how often does a randomly chosen image that has been labelled as containing a cat actually contain a cat? The steps to construct this metric are as follows. First, we set a threshold score such that when the algorithm is shown two randomly-chosen images — one that has a score greater than the threshold (i.e. a picture labelled as containing a cat) and another from those pictures that really does contain a cat — the probability that the image with the highest score is the one chosen from the set of real cat images is 50%. At this decision threshold, the set of positively labelled images are indistinguishable from the set of images which are positive. Then, as a second step, we measure performance by asking how often a randomly chosen picture from those labelled as containing a cat actually contains a cat. This metric can be thought of as *precision at the indistinguishability threshold*. While this new metric doesn't address the tradeoff between precision and recall inherent to all such metrics, I do show why this method avoids pitfalls that can occur when using, for example AUC, and it is better motivated than, for example, the F1-score.

![Illustration of the method](Article/Figures/FindCats.png)

## Code

`simple_example.py` is the example from figure 1.

`ABC.py` creates the rest of the figures in the article and saves them to `Article/Figures/`. It also prints the values of $C(r_b)$ quoted in the text. Set `include_other_curves=True` to produce `Bplot2.pdf` (figure 6) instead of `Bplot.pdf`. It requires `numpy`, `scikit-learn` and `matplotlib`.

## Article

The LaTeX source, bibliography and figures are in `Article/`, along with the compiled article, [`Article/ABC.pdf`](Article/ABC.pdf). `Article.zip` contains the source files.

### Changes since the arXiv version (v1, November 2023)

Section 3.6 (now *Relationship to F1 score*) replaces the earlier informal comparison with exact results. $B(r)$ decomposes into precision, recall and an AUC computed on the labelled members only. At the indistinguishability threshold, FN/FP equals $(2\alpha-1)C$. As a result, $r_b$ always lies below the precision–recall break-even point, and recall and F1 at $r_b$ follow from precision and $\alpha$. The section also shows why the F1-maximising threshold coincides with $r_b$ only in special cases.

The version here also corrects the following errors in the arXiv version.

- **Figures.** `ABC.py` used the wrong data set when computing $B(r)$ for six of the nine panels (it indexed the data with `cj*3+cj` instead of `cj*3+ci`). This moved $r_b$ in figures 2, 4, 5 and 6, most visibly in panels (g) and (h). The figures have been regenerated, and the values quoted in the text updated accordingly ($C(r_b) \approx 0.87$, $0.70$ and $0.53$ for the three columns of data sets).
- **Figure 1.** The worked example gave $B(65) \approx 0.49$ and $B(75) \approx 0.49$. With the scores shown, $B(67) = 18/37 \approx 0.49$ and $B(75) = 6/20 = 0.30$. Also, "real score score" is now "real score".
- **Equations.** The denominators in equations (10), (15), (18) and (19) should be $P v(r) + N u(r)$; the integrand in equations (14), (15) and (19) should be $f_N(t) v(t)$; and the integrals in equations (11) and (12) had incorrect limits or variables. The code always used the correct expressions.
- **Text.** The number of 'easy' data points per panel was listed in the wrong order; the description of what happens at large and small thresholds had "former" and "latter" swapped; and the logistic function should be $1/(1+\exp(-r))$. A number of smaller typos have also been fixed.
