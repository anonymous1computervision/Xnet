from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *


class Trainer(object):

    session = None

    def __init__(self, config):
        print("Defining Trainer ...")
        self.config = config
        self.logdir = config.train.logdir
        self.num_epochs = config.train.num_epochs
        self.init_lr = config.train.init_lr
        self.lr_decay = config.train.lr_decay
        self.print_epoch = config.train.print_epoch

    def train_and_evaluate(self, model, data_model, verbose=False):
        """Helper to run the model with different training modes."""
        print("Training ...")
        with tf.Session(graph=model.graph) as self.sess:
            tf.global_variables_initializer().run()
            summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir=self.logdir, graph=self.sess.graph)

            for i in range(self.num_epochs):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / self.num_epochs
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = self.init_lr / (1. + 10 * p) ** self.lr_decay

                X0, y0 = next(data_model.gen_source_batch)
                X1, y1 = next(data_model.gen_target_batch)
                y = np.vstack([y0, y1])
                train_feed_dict = {
                    model.Xm: X0,
                    model.Xmm: X1,
                    model.y: y,
                    model.domain: data_model.domain_labels,
                    model.train: True,
                    model.training: True,
                    model.l: l,
                    model.learning_rate: lr
                }
                _, batch_loss, dloss, ploss, d_acc, p_acc, summary = self.sess.run(
                    model.train_ops + [summary_op],
                    feed_dict=train_feed_dict)

                writer.add_summary(summary, i)

                if verbose and i % self.print_epoch == 0:
                    print('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                        batch_loss, d_acc, p_acc, p, l, lr))

            dann_acc_feed_dict = {
                model.Xm: data_model.mnist_test,
                model.Xmm: data_model.mnistm_test,
                model.y: data_model.test_labels,
                model.train: False,
                model.training: False}

            test_domain_acc_feed_dict = {
                model.Xm: data_model.test_imgs_m,
                model.Xmm: data_model.test_imgs_mm,
                model.domain: data_model.combined_test_domain,
                model.l: 1.0,
                model.training: True
            }
            test_emb_feed_dict = {
                model.Xm: data_model.test_imgs_m,
                model.Xmm: data_model.test_imgs_mm,
                model.training: True
            }

            dann_acc = self.sess.run(model.label_acc, feed_dict=dann_acc_feed_dict)
            test_domain_acc = self.sess.run(model.domain_acc, feed_dict=test_domain_acc_feed_dict)
            test_emb = self.sess.run(model.feature, feed_dict=test_emb_feed_dict)

            return dann_acc, test_domain_acc, test_emb

