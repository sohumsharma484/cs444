## CS 444: Deep Learning for Computer Vision, Fall 2024, Assignment 1

### Instructions
1.  Assignment is due at **11:59:59 PM on Thursday Sep 19 2024**.

2.  See [policies](https://saurabhg.web.illinois.edu/teaching/cs444/fa2024/policies.html)
    on [class website](https://saurabhg.web.illinois.edu/teaching/cs444/fa2024).

3.  Submission instructions:
    1.  On gradescope assignment (Gradescope Code **PYKW6R**) called `MP1-code`, upload your completed
    `models.py` and `featurize.py` files. These will be autograded and you will
    receive a score for your code.
        - Do not compress the files into `.zip` as this will not work.
        - Do not change the provided files names nor the names of the functions but
        rather change the code inside the provided functions and add new functions.
        Also, make sure that the inputs and outputs of the provided functions are
        not changed.
        - The autograder will give you feedback on how well your code did.
        - The autograder is configured with the python libraries noted in 
        `requirements.txt`. Autograding will fail if you use any packages that are not listed in requirements.txt and are not included by default with python.

    2. On gradescope assignment called `MP1-report`, fill out the text response
    to questions along with supporting figures and plots.

    3. We reserve the right to take off points for not following submission
    instructions. 

4.  Be careful not to work of a public fork of this repo. Make a
    private clone to work on your assignment. You are responsible for
    preventing other students from copying your work. Please also see point 2
    above.

5.  See [SUGGESTIONS.md](./SUGGESTIONS.md) for some suggestions for setup,
    workflow, and frequently asked questions.

### Problems

1. **Linear Algebra Review [4 pts Manually Graded].** Answer the following questions about
    matrices. Show the calculation steps (as applicable) to get full credit.

    1.1  **Matrix Multiplication [1 pts].** Let $`V = 
            \begin{bmatrix}
            -\frac{\sqrt 3}2 & \frac12 \\
                -\frac12 & -\frac{\sqrt 3}2
            \end{bmatrix}`$ Compute
        $`V \begin{bmatrix} 1 \\ 0 \end{bmatrix}`$ and  
        $`V \begin{bmatrix} 0 \\ 1 \end{bmatrix}`$. What does matrix
        multiplication $`Vx`$ do to $`x`$?


    1.2  **Matrix Transpose [1 pts].**  Verify that $`V^{-1} = V^\top`$ What does
        $`V^\top x`$ do to $`x`$?


    1.3  **Diagonal Matrix [1 pts].**  Let $`\Sigma = 
            \begin{bmatrix}
                3 & 0 \\
                0 & 5
            \end{bmatrix}`$ Compute $`\Sigma V^\top x`$ where
        $`x = \begin{bmatrix} \frac1{\sqrt{3}} \\ 0 \end{bmatrix}, 
            \begin{bmatrix} 0 \\ \frac1{\sqrt{3}} \end{bmatrix}, 
            \begin{bmatrix} -\frac1{\sqrt{3}} \\ 0 \end{bmatrix}, 
            \begin{bmatrix} 0 \\ -\frac1{\sqrt{3}} \end{bmatrix}`$ respectively.
        These are 4 corners of a square. How is the square transformed by $`\Sigma V^\top`$ ?

    1.4  **Geometric Interpretation [1 pts].**  Compute $`A = U\Sigma V^T = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 5 \end{bmatrix} \begin{bmatrix} -\frac{\sqrt 3}2 & \frac12 \\ -\frac12 & -\frac{\sqrt 3}2 \end{bmatrix}^\top`$ From the above
        questions, we can see a geometric interpretation of $`Ax`$: (1) $`V^\top`$
        first rotates point $`x`$ (2) $`\Sigma`$ rescales it along the
        coordinate axes, (3) then $`U`$ rotates it again. Now consider a
        general squared matrix $`B \in \mathbb{R}^{n\times n}`$ How would you obtain
        a similar geometric interpretation for $`Bx`$?

2. **Nearest Neighbors Classifier.**

    2.1 **[2 pts Autograded].** Complete the implementation of the
    `NearestNeighbor` class in [models.py](models.py). We will use L2 distance
    as the distance metric. You will need to complete
    the `train` and the `predict` functions. The input to the `predict` function
    is a *batch* of data that we want to make predictions for. 
    You are not allowed to use any external libraries or
    built-in functions that directly solve the problem for you (e.g. it is not
    ok to use `sklearn.neighbors.KNeighborsClassifier`, `cv2.knnMatch`, or
    `scipy.spatial.distance.cdist` among others). 
    
    *Hyperparameter selection:* The number of nearest neighbors `k` is a 
    hyper-parameter that you will need to select. Your implementation should 
    be general and handle different values fo `k` correctly. We found `k` values 
    of 1, 3 and 5 to be reasonably performant across the different settings. 

    *Testing your implementation:* You can test your implementation by running 
    the following command. This will use your implementation to make predictions 
    on the validation set and report the accuracy. 
    ```
    # Our implementation achieves an accuracy of about 64.07% on the validation
    # set in the following setting and takes about 2 seconds to run on a 4-core
    # 3.4GHz machine. 
    python demo.py --classifier knn --k 5 --num_train 100 --out_dir runs/knn-raw/
    
    # Our implementation achieves an accuracy of about 85.79% on the validation
    # set in the following setting and takes about 3 seconds to run on a 4-core
    # 3.4GHz machine. 
    python demo.py --classifier knn --k 5 --num_train 1000 --out_dir runs/knn-raw/
    ```
    Once you are happy with your implementation, you can run,
    ```
    python -m unittest tests.TestClassifier.test_knn_small 
    python -m unittest tests.TestClassifier.test_knn 
    ```
    This will evaluate your code on a number of other settings and confirm the
    results against the accuracies obtained by our implementation. We will be 
    running variants of these tests for autograding your code.

    *Hints:* Obvious ways to implement the `predict` function, say by using a
    for loop to go over the entire train / validation set, can be very slow. 
    You will benefit by vectorizing your code and avoiding any for loops to the
    extent possible. See [this tutorial](https://cs231n.github.io/python-numpy-tutorial/) to learn more about math
    and broadcasting operations on numpy objects. These are in-general
    significantly faster than writing for loops to iterate over the elements in
    a matrix. 
    
    At the same time naive vectorization that computes an intermediate
    $n_{train} \times n_{val} \times d$, where $d$ is the feature dimension (784
    in this case), will take too much memory. You will benefit from the
    following identity: $||a-b||^2_2 = (a-b)^T(a-b) = a^Ta + b^Tb -2a^tb$ and
    using the `np.dot` function. We would suggest to first focus on correctness
    (get the small tests, by running `python -m unittest
    tests.TestClassifier.test_knn_small` to pass first) and then worry about
    speed and memory. You can use your correct but inefficient code to confirm
    that your more efficient code works correctly.
    
    2.2 **[1 pts Manual Grading].** Next, we will visualize the nearest 
    neighbors for some sample digits. Complete the function 
    `get_nearest_neighbors` in `NearestNeighbor` to return the digit images
    corresponding to the  nearest neighbors from the training set for given
    test digit. You can use the plotting function `visualize_knn` in 
    [utils.py](utils.py) to visualize the nearest neighbors. Visualize the 10
    nearest neighbors for 5 random samples from the validation dataset in 2
    settings: a) when training dataset only has 100 samples and b) when training
    dataset has 10000 samples. Include the two generated visualization in
    submission to `MP1-report` and discuss what you observe.
    

3. **Linear Classifier.**

    3.1 **[2 pts Manual Graded].** In this problem, we will build multi-class 
    logistic regression classifiers to classify MNIST digits. Given a feature
    point $x$, the multi-class logistic regression classifier predicts the
    probability for each class $c$ as:
    $$p_c(x) = \frac{e^{w_c^Tx}}{\sum_{c'} e^{w_{c'}^Tx}} $$
    
    where $w_c$ is the parameter for class $c$. The classifier predicts the
    class with the highest probability. The parameters $w_c$ are learned by
    minimizing the cross-entropy loss as described below. Here, $x_i$ is the 
    data point from the training set and $y_i$ denotes the class it belongs to.
    $$L_d = -\frac{1}{N}\sum_{i=1}^N \log \left(p_{y_i} (x_i)\right)$$ 

    Typically, a regularization term, $L_r$, is also added to the loss function:
    $$L_r = \frac{\lambda}{2} \sum_{c=1}^C w_{c}^Tw_{c}$$

    where $\lambda$ is the regularization strength

    The total loss is then given by:
    $$L = L_d +  L_r$$
    
    Show that the gradient of $L_d$ with respect to $w_j$ is given by
    
    $$\frac{\partial L_d}{w_j} = \frac{1}{N} \sum_{i=1}^N (p_j(x_i) - \delta_{y_i, j}) x_i$$
    
    where $\delta_{y_i, j}$ is the Kronecker delta. Show your derivation in your answer.
    
    

    3.2 **[4 pts Autograded].** Complete the multi-class logistic regression implementation of the `train`,
    `predict` and the associated helper functions in `LinearClassifier` class in
    [models.py](models.py). You will need to write code to compute the loss
    function and its gradient with respect to the parameters
    (`compute_loss_and_gradient`). You will also need to complete a training
    loop (`train`) that uses gradient descent to optimize the parameters. You
    are not allowed to use any external libraries or built-in functions that
    directly solve the problem for you (e.g. it is not ok to use
    `sklearn.linear_model.LogisticRegression` or `pytorch`).

    *Hyperparameter selection:* The regularization strength $\lambda$ is a 
    hyper-parameter that you can play with. We found a value around 1e-4 to work
    reasonably well across the different settings. $\lambda$ is typically varied
    in orders of magnitude (e.g. 1e-4, 1e-3, 1e-2, etc.).

    *Testing your implementation:* We have included a) some unit tests and 
    b) end-to-end performance metrics to help you test your implementation. 
    For the unit test, you can run the following command:
    ```
    # python -m unittest tests.TestClassifier.test_gradient_and_loss -v 
    ```

    For the end-to-end tests, you can run the following command:
    ```
    # Takes 5 seconds and achieves an accuracy of 71.28%.
    python demo.py --classifier linear --lr 1e-1 --wt 1e-3 --num_train 100 \
        --out_dir runs/linear-raw/

    # Takes about 45 seconds to run on a 4-core 3.4GHz machine and achieves an accuracy of about 86%. 
    python demo.py --classifier linear --lr 1e-1 --wt 1e-3 --num_train 1000 \
        --out_dir runs/linear-raw/
    ```
    We also provide
    reference loss values as a function of the number of iterations for the above two runs. You can 
    access these using TensorBoard, using the following command: `tensorboard --logdir runs/linear-raw-reference/` and opening up the link that shows up. 

    Once you are happy with your implementation, you can run the following for a
    complete set of tests.
    ```
    python -m unittest tests.TestClassifier.test_linear
    ```
    *Hint:* We will again suggest to first focus on correctness (get the 
    unit test to pass first), and then work on optimizing your code. You can then compare the results of the two implementations to debug your optimized code.
    
    When computing the softmax, make sure you're using the [numerically stable softmax](https://jaykmody.com/blog/stable-softmax/)  so the intermediate values don't overflow.

    When computing the gradient, there are two important cases, When the class is the true class, then you'll want to update the weights for that class differently from when the class is not the true class. I would recommend working out what the derivative is by hand using the chain rule and then trying to implement it with simple for loops before vectorizing.

    3.3 **[1 pts Manual Grading].** Next, we will visualize the weights of the
    learned linear classifier. `demo.py` already saves the visualization of the
    learned weights to the specified out_dir folder as `w_vis.png`. Include the
    generated visualization in submission to `MP1-report` and discuss what you
    see. Do the weights correspond to the average shapes of the digits? Why or
    why not?

4. **Pooled Pixel Features. [3 pts Autograded]** Next, we will implement
    *simplified* version of features described in [this tech
    report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-159.html)
    on digit classification. We will start with a simple feature that pools the
    pixels in an image. Complete the `pool` function in
    [featurize.py](featurize.py). The input to this function is a batch of
    images and the output is the pooled features for each batch of images. 
    
    The pooled feature vector computes the *mean* of pixel values within the
    pooling region. We will tile the 28 x 28 MNIST digit images with
    non-overlapping `pool_size x pool_size` regions and compute a single scalar
    for each pooling region. So, if we are using a `pool_size` of 7, we will end
    up with a $4 \times 4 = 16$ dimensional feature vector for each digit. Fill
    up code in the `pool` function to compute this feature vector.  You may
    find the `np.bincount` function useful.

    *Testing your implementation:* You can check your implementation by training
    nearest neighbors and linear classifiers on this feature vector instead of
    raw pixels as features, using the following commands:
    
    ```
    # For knn classifier
    # Following ran in 3 seconds and gave 88% accuracy.
    python demo.py --classifier knn --k 1 --feature pool --pool_size 2 \
        --num_train 1000 --out_dir runs/knn-pool2/
    
    # For linear classifier
    # Following ran in 15 seconds and gave 86.54% accuracy.
    python demo.py --classifier linear --lr 1e-1 --wt 1e-3 --feature pool \
        --pool_size 2 --num_train 1000 --out_dir runs/linear-pool2/
    ```

    *Hyperparameter selection:* In addition to the classifier hyper-parameters 
    (e.g. `k` for nearest neighbors, $\lambda$ for linear classifier), you will
    also need to select the pooling region size. We found a pooling region size
    of 2 to work reasonably well across the different settings.
    
    Once you are happy with your implementation, you can run,
    ```
    python -m unittest tests.TestClassifier.test_knn_pool
    python -m unittest tests.TestClassifier.test_linear_pool
    ```
    This will evaluate your code on a number of other settings and confirm the
    results against the accuracies obtained by our implementation. We will be 
    running variants of these tests for autograding your code.

5. **Bias Variance Trade-offs. [3 pts Manually Graded].** Plot the validation
   accuracy for the 2 different features (raw pixels, pooled pixels) as a
   function of the amount of training data. Include the plots in your
   submission to `MP1-report`. Do this
   separately for both the linear classifier and the nearest neighbor
   classifier. Vary the number of training points as 100, 200, 500, 1000, 2000, 
   5000, 10000. Ideally, you will need to do separate validation for each of these
   settings and report the best accuracy across the hyper-parameters. However,
   for the purpose of this assignment you can ignore this and simply use the
   same hyper-parameters as noted above for all the settings. You may find
   [collect_and_plot.py](collect_and_plot.py) useful, but please adapt it as
   necessary. Running with larger number of training points will take a while. 
   You will benefit if your code is vectorized and efficient. It took us about 
   an hour of compute to get these plots.

6. **Histogram of Gradient Features. [2pts Autograded, Extra Credit]** Next, we will
    implement a simplified version of histogram of oriented gradient features. 
    This involves:
    1. Computing the gradient of the image in the x and y directions. You can do
    this by simply subtracting the consecutive pixels in the x and y
    directions. You can pad the original image (or the gradient image) with a
    row (or a column) of zeros to make the gradients the same size as the
    original image.
    2. Computing the magnitude and orientation of the gradient. You may find `np.arctan2` function useful.
    3. Compute the sum of gradient magnitude in the different orientations in the 
    different pooling regions on the image. You may find the function `np.bincount` useful.
    4. Concatenating the histograms from different pooling regions to form the
    full feature vector.

    Note that this is a simpler version of the features described in the [tech
    report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-159.html). 
    The simpler version will suffice for the programming assignment.

    Complete the implementation in the `hog` function in
    [featurize.py](featurize.py). The input to this function is a batch of
    images and the output is the HOG features for each batch of images. You may
    find the functions `np.pad`, `np.bincount` useful.

    *Hyperparameter selection:* The feature specific hyper-parameters are the 
    number of angle bins and the pooling region size. We experimented with 18
    bins for signed gradients and found it to work well. For pooling size,
    values of 7 and 4 were most effective for knn and linear classifier
    respectively.

    *Testing your implementation:* As before you can run your implementation using
    the following commands:
    ```
    # For knn classifier
    # Following ran in 3 seconds and gave 89.03% accuracy.
    python demo.py --classifier knn --k 1 --feature hog --pool_size 7 \
        --num_train 1000 --out_dir runs/knn-hog7/
    
    # For linear classifier
    # Following ran in 25 seconds and gave 93.96% accuracy.
    python demo.py --classifier linear --lr 1e-1 --wt 1e-3 --feature hog \
        --pool_size 4 --num_train 1000 --out_dir runs/linear-hog4/
    ```
    You can run the complete suite of tests using:

    ```
    python -m unittest tests.TestClassifier.test_knn_hog
    python -m unittest tests.TestClassifier.test_linear_hog
    ```

