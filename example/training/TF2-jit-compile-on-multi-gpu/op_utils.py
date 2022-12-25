import tensorflow as tf
from train import compute_sparse_kernel, sparsify

def Optimizer(args, model, strategy):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
    if args.spgrad is not None:
        optimizer = tf.keras.optimizers.SGD(args.learning_rate, 0, nesterov=False)
    else:
        optimizer = tf.keras.optimizers.SGD(args.learning_rate, .9, nesterov=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function(experimental_compile = args.compile)
    def compiled_step(images, labels):
        with tf.GradientTape() as tape:
            pred = model(images, training = True)
            total_loss = loss_object(labels, pred)/args.batch_size
        gradients = tape.gradient(total_loss, model.trainable_variables)
        update_vars = [model.Layers[k].update_var if hasattr(model.Layers[k], 'update_var') else None for k in model.Layers ]
        return total_loss, pred, gradients, update_vars

    def train_step(epoch, images, labels):
        total_loss, pred, gradients, update_vars = compiled_step(images, labels)
        if args.weight_decay > 0.:
            gradients = [g+v*args.weight_decay for g,v in zip(gradients, model.trainable_variables)]

        if epoch >= int(args.spg_epoch):
            if args.spgrad is not None :
                N, M = args.spgrad
                N = int(N)
                M = int(M)

                '''
                print("gradients:", len(gradients), "N:", N, "M:", M)
                for g,v in zip(gradients, model.trainable_variables):
                    if 'kernel:0' in v.name:
                        print("name:", v.name)
                        print("\t", g, v)
                        #print("\tbefore:", g[0,0,0,:].numpy())
                        #print("after:", compute_sparse_kernel(g, N, M)[0,0,0,:])
                '''
                gradients = [compute_sparse_kernel(g, N, M) if 'kernel:0' in v.name else g for g,v in zip(gradients, model.trainable_variables) ]

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for k, v in zip(model.Layers, update_vars):
            if hasattr(model.Layers[k], 'update'):
                model.Layers[k].update(v)

        if epoch >= int(args.spg_epoch):
            if args.spratio is not None:
                N, M = args.spratio
                N = int(N)
                M = int(M)
                #sparsify(model, N, M)
                for v in model.trainable_variables:
                    #print(v.name)
                    if 'kernel:0' in v.name:
                        #print(v.shape)
                        sparse_kernel = compute_sparse_kernel(v, N, M)
                        v.assign(sparse_kernel)


        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, pred)

    @tf.function
    def train_step_dist(epoch, image, labels):
        strategy.run(train_step, args= (epoch, image, labels))

    return train_step_dist, train_loss, train_accuracy, optimizer

