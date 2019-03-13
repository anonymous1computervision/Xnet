from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *
from flip_gradient import flip_gradient


class YNet(object):
    """
    MNIST-MNIST-M domain adaptation model.
    """
    logits = None
    domain_logits = None
    graph = None
    train_ops = None
    label_acc = None
    domain_acc = None

    def __init__(self, config, data_stats):
        print("Initializing YNet ...")
        self.config = config
        self.batch_size = self.config.train.batch_size

        self._define_train_ops(data_stats)

    def _dcca_correlate(self, Xm_input, Xmm_input):
        with tf.variable_scope('dcca_correlater'):
            W_convm0 = weight_variable([3, 3, 3, 4])
            b_convm0 = bias_variable([4])
            h_convm0 = tf.nn.relu(conv2d(Xm_input, W_convm0) + b_convm0)

            W_convm1 = weight_variable([3, 3, 4, 8])
            b_convm1 = bias_variable([8])
            h_convm1 = tf.nn.relu(conv2d(h_convm0, W_convm1) + b_convm1)

            W_convmm0 = weight_variable([3, 3, 3, 4])
            b_convmm0 = bias_variable([4])
            h_convmm0 = tf.nn.relu(conv2d(Xmm_input, W_convmm0) + b_convmm0)

            W_convmm1 = weight_variable([3, 3, 4, 8])
            b_convmm1 = bias_variable([8])
            h_convmm1 = tf.nn.relu(conv2d(h_convmm0, W_convmm1) + b_convmm1)

            return h_convm1, h_convmm1

    def _extract_featrue(self, X_input):
        with tf.variable_scope('feature_extractor'):
#            self.mid_1 = tf.reshape(X_input, [-1, 28 * 28 * 8])
            W_conv0 = weight_variable([3, 3, 8, 16])
            b_conv0 = bias_variable([16])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

#            self.mid_1 = tf.reshape(h_pool0, [-1, 14 * 14 * 16])

            W_conv1 = weight_variable([3, 3, 16, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            self.mid_2 = tf.reshape(h_pool1, [-1, 7 * 7 * 32])

            W_conv2 = weight_variable([3, 3, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            # The domain-invariant feature
            self.feature = tf.reshape(h_conv2, [-1, 7 * 7 * 64])

    def _predict_label(self, h_1):
        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [self.batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [self.batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([7 * 7 * 64, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2
#            Uncomment this to change to dual adversarial mode            
#            logits_g = attention_module(h_1, logits)
#            self.pred = tf.nn.softmax(logits)
#            self.logits = logits
            self.pred = tf.nn.softmax(logits)
            self.logits = logits

    def _predict_domain(self):
        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            feat_2 = flip_gradient(self.mid_2, self.l)
            d_W_fc2 = weight_variable([7 * 7 * 32, 100])
            d_b_fc0 = bias_variable([100])
            d_total = tf.matmul(feat_2, d_W_fc2) + d_b_fc0
            d_h_fc0 = tf.nn.relu(d_total)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_logits = d_logits

    def _define_model(self, data_stats):
        mnist_mean, mnistm_mean = data_stats
        Xm_input = (tf.cast(self.Xm, tf.float32) - mnist_mean) / 255
        Xmm_input = (tf.cast(self.Xmm, tf.float32) - mnistm_mean) / 255

        h_convm1, h_convmm1 = self._dcca_correlate(Xm_input, Xmm_input)
        X_input = tf.cond(self.training, lambda: tf.concat([h_convm1, h_convmm1], 0), lambda: h_convmm1)


        self._extract_featrue(X_input=X_input)
        self._predict_label(h_1 = h_convmm1)
        self._predict_domain()

    def _define_loss(self):
        self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.classify_labels)
        self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.domain_logits, labels=self.domain)

    def _define_placeholders(self):

        self.Xm = tf.placeholder(tf.uint8, [None, 28, 28, 3], name='Xm')
        self.Xmm = tf.placeholder(tf.uint8, [None, 28, 28, 3], name='Xmm')
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3], name='X')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.domain = tf.placeholder(tf.float32, [None, 2], name='domain')
        self.l = tf.placeholder(tf.float32, [], name='l')
        self.training = tf.placeholder(tf.bool, name='training') # for whether concat
        self.train = tf.placeholder(tf.bool, [], name='train')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        tf.summary.scalar("learning_rate", self.learning_rate)

    def _define_train_ops(self, data_stats):
        print("Defininig Train ops ...")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._define_placeholders()
            self._define_model(data_stats)
            self._define_loss()

            pred_loss = tf.reduce_mean(self.pred_loss)
            domain_loss = tf.reduce_mean(self.domain_loss)
            total_loss = pred_loss + domain_loss 
            tf.summary.scalar("pred_loss", pred_loss)
            tf.summary.scalar("domain_loss", domain_loss)
            tf.summary.scalar("total_loss", total_loss)

            dann_train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.65).minimize(total_loss)

            # Evaluation
            correct_label_pred = tf.equal(tf.argmax(self.classify_labels, 1), tf.argmax(self.pred, 1))
            label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
            correct_domain_pred = tf.equal(tf.argmax(self.domain, 1), tf.argmax(self.domain_pred, 1))
            domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
            self.train_ops = [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc]
            self.label_acc = label_acc
            self.domain_acc = domain_acc
