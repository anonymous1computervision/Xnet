# Xnet

This is a implement of Xnet. We take the transfer between Mnist and Mnist-M as an example. 

The final result would be 0.892 after 10k epochs, 0.910 after 40k epochs(Domain accuracy: 0.569), 0.963 after around 0.6m epochs(Domain accuracy: 0.547). Different times of implementations might result in trivial different accuracies. If you want to stablize the result, add seed in your forked repo. 

# Run Training

```
python train.py
```


# See logs

```
tensorboard --logdir=log
```

# Environment

All code was tested on Ubuntu 16.04 with Python 2.7. You also need the following package:
tensorflow
cv2
python-box
ruamel.yaml

The further optimized code will be uploaded when the paper published.
