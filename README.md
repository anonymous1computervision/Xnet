# Xnet

This is a implement example of Xnet. We take the transfer between Mnist and Mnist-M as an example. 

The final result would be 0.892 after 10k epochs, 0.910 after 40k epochs(Domain accuracy: 0.569), 0.963 after around 0.6m epochs(Domain accuracy: 0.547). Different times of implementations might result in trivial different accuracies. If you want to stablize the result, add seed in your forked repo. 

The "flip_gradient.py" is forked from https://github.com/pumpikano/tf-dann. We would thanks so much to the authors of this repo to provide such fabulous open source code.


# Run Training

```
python train.py
```


# See logs

```
tensorboard --logdir=log
```
# data benchmark
Here we put all the Mnist and Mnist-M data in this repo. Just download the whole repo and run. 

All the six data benchmarks we collected and preprocessed will be released when the paper published.

# Environment

All code was tested on Ubuntu 16.04 with Python 2.7. You also need the following package:
tensorflow
cv2
python-box
ruamel.yaml

More further optimized code will be uploaded when the paper published.
