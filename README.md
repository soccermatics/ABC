# The ABC test

This repository contains the code and article for 'A straightforward method for evaluating performance of machine learning models for classification'

Imagine you have created an algorithm which assigns a score to a photograph based on how likely it is to contain a cat.  I propose the following steps to test performance of your model. First, you should set a threshold score such that when the algorithm is shown two randomly-chosen images --- one that has a score greater than the threshold (i.e. a picture labelled as containing a cat) and another from those pictures that really does contain a cat--- the probability that the image with the highest score is the one chosen from the set of real cat images is 50\%. This method thus sets a decision threshold, so that the set of positively labelled images are indistinguishable from the set of images which are positive. Once this threshold is established, as a second step, we measure performance by asking how often a randomly chosen picture from those labelled as containing a cat actually contains a cat. This measure is generally known as the precision, but we evaluate it specifically for the threshold chosen at the first step. The result is a single number measuring the performance of the algorithm. I explain, using an example, why this method avoids pitfalls inherent to, for example AUC, and is better motivated than, for example, the F1-score.

![](Article/Figures/Find Cats.pdf)

## Code

simple_example.py is the example from figure 1.

ABC.py creates the rest of the figures from the article. They are output in Article/Figures/

## Article

The article is provided in this directory in latex, along with the Figures

