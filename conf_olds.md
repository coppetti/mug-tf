

# Home Test 01
# configs = {
#     "features": [-1,64,64,3],
#     "mean": 0,
#     "stddev": 0.08,
#     "strides": [1, 1, 1, 1],
#     "pool_strides": [1, 2, 2, 1],
#     "ksize": [1, 2, 2, 1],
#     "num_outputs": 1024,
#     "logit_outputs": 4,
#     "prob": 0.9,
#     "padding": "SAME",
#     "pred_class_axis": 1,
#     "pred_learn_rate": 0.01,
#     "batch_size": 75,
#     "training_steps": 300
# }

# Home Test 02
# configs = {
#     "features": [-1,64,64,3],
#     "mean": 0,
#     "stddev": 0.08,
#     "strides": [1, 1, 1, 1],
#     "pool_strides": [1, 2, 2, 1],
#     "ksize": [1, 2, 2, 1],
#     "num_outputs": 256,
#     "logit_outputs": 4,
#     "prob": 0.9,
#     "padding": "SAME",
#     "pred_class_axis": 1,
#     "pred_learn_rate": 0.01,
#     "batch_size": 75,
#     "training_steps": 350
# }



    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits    = logits)



        # train_op1 = tf.train.AdagradOptimizer(learning_rate=configs["pred_learn_rate"]).minimize(loss= loss, global_step = tf.train.get_global_step())
        # train_op2 = tf.train.GradientDescentOptimizer(learning_rate=configs["pred_learn_rate"]).minimize(loss= loss, global_step = tf.train.get_global_step())
        # train_op3 = tf.train.MomentumOptimizer(learning_rate=configs["pred_learn_rate"], momentum=0.9, use_nesterov=False).minimize(loss=loss, global_step=tf.train.get_global_step())
        # train_op= tf.group( train_op3, train_op1)
