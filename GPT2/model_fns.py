from functools import partial

import numpy as np
import tensorflow as tf

from optimizers import create_train_op
from metric_fns import *

from models.gpt2.sample import make_sect_and_center

def gpt2_model(features, labels, mode, params):
    from models.gpt2 import gpt2

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

            output["pred"] = tf.cast(output["pred"], tf.float32)

        else:
            output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

        # loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output["logits"], labels=labels)
        # loss_batch = tf.nn.l2_loss(output["pred"] - labels)
        n_bins = params['multibin_nbins']
        overlap = params['multibin_overlap']
        s_min = params['multibin_min']
        s_max = params['multibin_max']
        sections, centers = make_sect_and_center(
            n_bins=n_bins,
            overlap=overlap,
            span=(s_min, s_max))
        tfsections = tf.constant(sections)
        tfcenters = tf.constant(centers)
        cls_logit = output["pred"][:,:,:,:,0]
        reg_val = output["pred"][:,:,:,:,1]
        
        best_class = tf.argmin(tf.abs(
            tfcenters[None,None,None] - tf.expand_dims(labels, -1)), axis=-1)
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=best_class,
                logits=cls_logit)
        bucket = tf.logical_and(
                tfsections[None,None,None,:,0] < labels[:,:,:,None],
                tfsections[None,None,None,:,1] > labels[:,:,:,None],)
        reg_loss = tf.multiply(
                tf.squared_difference(reg_val + tfcenters, labels[:,:,:,None]),
                tf.cast(bucket, tf.float32))
        loss_batch = cls_loss + tf.reduce_sum(reg_loss, axis=-1)
        loss = tf.reduce_mean(loss_batch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_train_op(loss, params)

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == tf.estimator.ModeKeys.EVAL:
        # from metric_fns import perplexity_metric

        if params["use_tpu"]:
            # Metric inputs are transferred to CPU and must preserve batch dimension
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                loss=loss)
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss)


    if mode == tf.estimator.ModeKeys.PREDICT:
        from models.gpt2 import sample

        if not "top_k" in params.keys():
            params["top_k"] = 0
        output = sample.sample_sequence(
            params=params, length=params["n_pred"],
            context=features,
            batch_size=params["batch_size"],
            temperature=1.0,
        )

        predictions = {
            "tokens": output
        }

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
