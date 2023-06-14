import tensorflow as tf
import numpy as np
   
class SummaryTracker():
    def __init__(self, name, summary_writer, average=True):
        self.name = name
        self.step = 1
        self.summary_writer = summary_writer
        self.tf_tracker = tf.metrics.Mean(name, dtype=tf.float32) if average else tf.metrics.Sum(name, dtype=tf.float32)

    def __call__(self, val):
        self.tf_tracker(val)
        if np.isfinite(val):
            with self.summary_writer.as_default():
                tf.summary.scalar(self.name, val, step=self.step)
        self.step += 1

    def result(self):
        return self.tf_tracker.result()
    
    def reset_states(self):
        self.tf_tracker.reset_states()