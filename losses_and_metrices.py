

import tensorflow as tf
import numpy as np

class compactnessLoss(tf.keras.losses.Loss):
    def __init__(self, name='compactness_loss'):
        super(compactnessLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):

        n_dim = np.shape(y_pred)[0]  # number of features vecs
        k_dim = np.shape(y_pred)[1]  # feature vec dim
        loss = tf.constant(0.0)
        for i in range(0, n_dim):
            mask = np.ones(n_dim)
            mask[i] = 0
            mask = tf.constant(mask)
            others = tf.boolean_mask(y_pred, mask)

            mean_vec = tf.math.reduce_sum(others, axis=0) / float(n_dim - 1)
            diff = tf.math.subtract(y_pred[i], mean_vec) / float(n_dim)
            loss = tf.math.add(tf.math.reduce_sum(tf.math.pow(diff, 2)), loss)
        return loss

class FeaturesLoss:
    def __init__(self, templates_images, model):
        self.templates_features = self.build_templates(templates_images, model)

    def build_templates(self, templates_images, model):
        templates = []
        for i in range(templates_images.shape[0]):
            image = np.expand_dims(templates_images[i], axis=0)
            templates.append(
                np.squeeze(model(image, training=False), axis=0))
        return np.array(templates)

    def __call__(self, labels, preds):
        preds_num = preds.shape[0]
        losses = np.zeros(preds_num)
        for i in range(preds_num):
            distances = []
            for t in range(self.templates_features.shape[0]):
                distances.append(np.sqrt(float(np.dot(preds[i] - self.templates_features[t],
                                                      preds[i] - self.templates_features[t]))))  # Eucleaden distance
            losses[i] = min(distances)
        return losses



class AOC_helper:
    @staticmethod
    def get_roc_aoc(tamplates, targets, aliens, model):
        fpr, tpr, thresholds, roc_auc, target_scores, alien_scores = AOC_helper.get_roc_aoc_with_scores(tamplates, targets, aliens, model)

        return fpr, tpr, thresholds, roc_auc, np.mean(target_scores), np.mean(alien_scores)

    @staticmethod
    def get_roc_aoc_with_scores(tamplates, targets, aliens, model):
        loss_func = FeaturesLoss(tamplates, model)

        target_num = len(targets)
        alien_num = len(aliens)

        scores = np.zeros(target_num + alien_num)
        labels = np.zeros(target_num + alien_num)

        preds = model(targets, training=False)
        scores[:target_num] = loss_func(None, preds)
        labels[:target_num] = np.zeros(target_num)

        preds = model(aliens, training=False)
        scores[target_num:] = loss_func(None, preds)
        labels[target_num:] = np.ones(alien_num)

        fpr, tpr, thresholds = roc_curve(labels, -scores, 0)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc, scores[:target_num], scores[target_num:]


