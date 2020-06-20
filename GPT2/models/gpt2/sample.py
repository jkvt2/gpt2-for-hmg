import tensorflow as tf
import numpy as np

from models.gpt2 import gpt2

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        tf.equal(k, 0),
        lambda: logits,
        lambda: _top_k(),
    )

def make_sect_and_center(n_bins, overlap, span):
    section_length = (span[1] - span[0])/(n_bins - (n_bins - 1) * overlap)
    sections = [span[0] + i * (1 - overlap) * section_length for i in range(n_bins)]
    sections = [[i, i+section_length] for i in sections]
    centers = [(i+j)/2 for i,j in sections]
    sections[0][0] = -np.inf
    sections[-1][1] = np.inf
    return sections, centers

def sample_sequence(*, params, length, start_token=None, batch_size=None, context=None, temperature=1, gaussian_noise=0, top_k=0):
    n_bins = params['multibin_nbins']
    overlap = params['multibin_overlap']
    s_min = params['multibin_min']
    s_max = params['multibin_max']
    sections, centers = make_sect_and_center(
        n_bins=n_bins,
        overlap=overlap,
        span=(s_min, s_max))
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    # length = length - params["text_len"]

    def step(params, tokens, past=None):
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                lm_output = gpt2.model(params=params, X=tokens, past=past, reuse=tf.AUTO_REUSE)

            lm_output["pred"] = tf.cast(lm_output["pred"], tf.float32)

        else:
            lm_output = gpt2.model(params=params, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        pred = lm_output['pred']
        presents = lm_output['present']
        presents.set_shape(gpt2.past_shape(params=params, batch_size=batch_size))
        return {
            'pred': pred,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):

        context_output = step(params, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(params, prev[:, tf.newaxis], past=past)
            pred = next_outputs['pred'][:, -1, :]
            logits = pred[:,:,:,0]
            resids = pred[:,:,:,1]
            logits = top_k_logits(logits, k=top_k) # batch_size x 85 x 21
            regressed = resids + tf.constant(centers)[None,None] # batch_size x 85 x 21
            sampled_bin = tf.multinomial(
                tf.reshape(logits, (-1, n_bins)), # (batch_size x 85) x 21
                num_samples=1,
                output_dtype=tf.int32) # (batch_size x 85) x 1
            sampled_bin = tf.reshape(sampled_bin, (-1, 85)) # batch_size x 85
            idxs = tf.reshape(sampled_bin, (-1,))
            idxs = tf.stack([tf.range(tf.size(idxs)), idxs], axis=-1)
            samples = tf.gather_nd(
                tf.reshape(regressed, (-1, n_bins)), # (batch_size x 85) x 21
                idxs) # (batch_size x 85)
            samples = tf.reshape(samples, (-1, 85))
            samples = samples + tf.random.normal(tf.shape(samples), stddev=gaussian_noise)
            past = tf.concat([past, next_outputs['presents']], axis=-2)
            prev = samples
            output = tf.concat([output, samples[:,None]], axis=1)
            return [past, samples, output,]

        def cond(*args):
            return True
        
        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(gpt2.past_shape(params=params, batch_size=batch_size)),
                tf.TensorShape([None, 85]),
                tf.TensorShape([None, None, 85]),
            ],
            back_prop=False,
        )

        return tokens
