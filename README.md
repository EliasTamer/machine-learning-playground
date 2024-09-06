# Machine Learning By Prof. Andrew Ng :star2::star2::star2::star2::star:

This page contains all my YouTube/Coursera Machine Learning courses and resources :book: by [Prof. Andrew Ng](http://www.andrewng.org/) :man:

# Table of Contents
1. [Brief Intro](#brief-intro)

## Brief Intro

The most of the course talking about **hypothesis function** and minimising **cost funtions**

### Hypothesis
A hypothesis is a certain function that we believe (or hope) is similar to the true function, the target function that we want to model. In context of email spam classification, it would be the rule we came up with that allows us to separate spam from non-spam emails.

### Cost Function
The cost function or **Sum of Squeared Errors(SSE)** is a measure of how far away our hypothesis is from the optimal hypothesis. The closer our hypothesis matches the training examples, the smaller the value of the cost function. Theoretically, we would like J(θ)=0

### Gradient Descent
Gradient descent is an iterative minimization method. The gradient of the error function always shows in the direction of the steepest ascent of the error function. Thus, we can start with a random weight vector and subsequently follow the
negative gradient (using a learning rate alpha)

#### Differnce between cost function and gradient descent functions
<table>
    <colgroup>
        <col width="50%" />
        <col width="50%" />
    </colgroup>
    <thead>
        <tr class="header">
            <th> Cost Function </th>
            <th> Gradient Descent </th>
        </tr>
    </thead>
    <tbody>
        <tr valign="top">
            <td markdown="span">
            <pre><code>
            function J = computeCostMulti(X, y, theta)
                m = length(y); % number of training examples
                J = 0;
                predictions =  X*theta;
                sqerrors = (predictions - y).^2;
                J = 1/(2*m)* sum(sqerrors);
            end
            </code></pre>
            </td>
            <td markdown="span">
            <pre><code>
            function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)    
                m = length(y); % number of training examples
                J_history = zeros(num_iters, 1);
                for iter = 1:num_iters
                    predictions =  X * theta;
                    updates = X' * (predictions - y);
                    theta = theta - alpha * (1/m) * updates;
                    J_history(iter) = computeCostMulti(X, y, theta);
                end
            end
            </code></pre>
            </td>
        </tr>
    </tbody>
</table>

### Bias and Variance
When we discuss prediction models, prediction errors can be decomposed into two main subcomponents we care about: error due to "bias" and error due to "variance". There is a tradeoff between a model's ability to minimize bias and variance. Understanding these two types of error can help us diagnose model results and avoid the mistake of over- or under-fitting.

Source: http://scott.fortmann-roe.com/docs/BiasVariance.html

### Hypotheis and Cost Function Table

| Algorithm 	| Hypothesis Function 	| Cost Function 	| Gradient Descent 	|
|--------------------------------------------	|-----------------------------------------------------------------------	|-------------------------------------------------------------------------------	|---------------------------------------------------------------------------------------	|
| Linear Regression 	| ![linear_regression_hypothesis](/images/linear_hypothesis.gif) 	| ![linear_regression_cost](/images/linear_cost.gif) 	|  	|
| Linear Regression with Multiple variables 	| ![linear_regression_hypothesis](/images/linear_hypothesis.gif) 	| ![linear_regression_cost](/images/linear_cost.gif) 	| ![linear_regression_multi_var_gradient](/images/linear_multi_var_gradient_descent.gif) 	|
| Logistic Regression 	| ![logistic_regression_hypothesis](/images/logistic_hypothesis.gif) 	| ![logistic_regression_cost](/images/logistic_cost.gif) 	| ![logistic_regression_gradient](/images/logistic_gradient.gif) 	|
| Logistic Regression with Multiple Variables 	|  	| ![logistic_regression_multi_var_cost](/images/logistic_multi_var_cost.gif) 	| ![logistic_regression_multi_var_gradient](/images/logistic_multi_var_gradient.gif) 	|
| Neural Networks 	|  	| ![nural_cost](/images/nural_cost.gif) 	|  	|                                                                                      |
