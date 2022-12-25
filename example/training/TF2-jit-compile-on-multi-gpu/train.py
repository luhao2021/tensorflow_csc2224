import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
tf.debugging.set_log_device_placement(False)

from dataloader import ILSVRC, CIFAR
import op_utils, utils, nets

parser = argparse.ArgumentParser(description='')
parser.add_argument("--train_path", default="test", type=str, help = 'path to log')
parser.add_argument("--data_path", default="E:/ILSVRC2012", type=str, help = 'home path for ILSVRC dataset')
parser.add_argument("--arch", default='ResNet-50', type=str, help = 'network architecture. currently ResNet is only available')
parser.add_argument("--dataset", default='ILSVRC', type=str, help = 'ILSVRC or CIFAR{10,100}')

parser.add_argument("--learning_rate", default = 1e-1, type=float, help = 'initial learning rate')
parser.add_argument("--decay_points", default = [.3, .6, .9], type=float, nargs = '+', help = 'learning rate decay point')
parser.add_argument("--decay_rate", default=.1, type=float, help = 'rate to decay at each decay points')
parser.add_argument("--weight_decay", default=1e-4, type=float, help = 'decay parameter for l2 regularizer')
parser.add_argument("--batch_size", default = 256, type=int, help = 'training batch size')
parser.add_argument("--val_batch_size", default=256, type=int, help = 'validation batch size')
parser.add_argument("--train_epoch", default=100, type=int, help = 'total training epoch')

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+', help = 'denote which gpus are used')
parser.add_argument("--do_log", default=200, type=int, help = 'logging period')
parser.add_argument("--compile", default=False, action = 'store_true', help = 'denote use compile or not. True is recommended in this repo')
parser.add_argument('--spratio', type=int, nargs='+', default=None, help='sparse ratio')
parser.add_argument('--spgrad', type=int, nargs='+', default=None, help='sparse ratio for gradient')
parser.add_argument('--spg_epoch', type=int, default=0, help='epoch to start sparsify')
parser.add_argument("--ckpt_name", default="ckpt0", type=str, help = 'checkpoint file name')
args = parser.parse_args()

# run deterministically
seed = 42
tf.random.set_seed(seed)


args.home_path = os.path.dirname(os.path.abspath(__file__))
args.decay_points = [int(dp*args.train_epoch) if dp < 1 else int(dp) for dp in args.decay_points]

if args.dataset == 'ILSVRC':
    args.weight_decay /= len(args.gpu_id)
    args.learning_rate *= args.batch_size/256

def compute_sparse_kernel(weights, N, M):
    weights_ = tf.reshape(weights, [-1, int(M)])
    mask_index = tf.argsort(tf.abs(weights_), axis=1)[:, :int(M-N)]
    indice_x = tf.reshape(tf.repeat(tf.range(mask_index.shape[0], dtype=tf.int32), int(M-N)), [-1,int(M-N)])
    indices = tf.reshape(tf.stack([indice_x, mask_index], axis=-1), [-1, 2])

    weight_mask = tf.ones(weights_.shape, dtype=tf.int32)
    weight_mask = tf.tensor_scatter_nd_update(weight_mask, indices, tf.zeros(indices.shape[0], dtype=tf.int32))

    sp_mask = tf.reshape(weight_mask, tf.shape(weights))
    sp_mask = tf.cast(sp_mask, dtype=tf.float32)
    sparse_kernel = tf.multiply(weights, sp_mask)

    return sparse_kernel

def sparsify(model, N, M):
    for i, layer in enumerate(model.layers):
        #print(i, type(layer))
        if type(layer) == nets.tcl.Conv2d or type(layer) == nets.tcl.FC:
            #print("layer:", i, type(layer), len(weights), tf.shape(weights[0]), tf.shape(weights[1]))
            weights = layer.get_weights()
            sparse_kernel = compute_sparse_kernel(weights[0], N, M)
            layer.set_weights([sparse_kernel, weights[1]])

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    if args.spratio is not None:
        N, M = args.spratio
        N = int(N)
        M = int(M)
        print("sparsity: (%d,%d)" % (N,M))

    with strategy.scope():
        if args.dataset == 'ILSVRC':
            datasets = ILSVRC.build_dataset_providers(args, strategy)
        elif 'CIFAR' in args.dataset:
            datasets = CIFAR.build_dataset_providers(args, strategy)
        model = utils.load_model(args, datasets['num_classes'])

        summary_writer = tf.summary.create_file_writer(args.train_path, flush_millis = 30000)
        with summary_writer.as_default():
            utils.save_code_and_augments(args)
            total_train_time = 0

            train_step, train_loss, train_accuracy, optimizer = op_utils.Optimizer(args, model, strategy )

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
            Eval = utils.Evaluation(args, model, strategy, datasets['test'], loss_object)

            best_acc = tf.Variable(0.0)
            curr_epoch = tf.Variable(0)
            ckpt_path = './checkpoints/' + args.ckpt_name
            ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc, optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)


            print ('Training starts')
            fine_tuning_time = 0
            tic = time.time()
            for step, data in enumerate(datasets['train']):
                epoch = step//datasets['train_len']
                lr = utils.scheduler(args, optimizer, epoch)
                train_step(epoch, *data)

                '''
                if args.spratio is not None:
                    sparsify(model, N, M)
                '''

                if args.spgrad is not None or args.spratio is not None:
                    print("step:%d, print the first 64 element of weight:" % (step))
                    #print(model.layers)
                    print(model.layers[4].get_weights()[0][0,0,0,:])

                step += 1
                # if 1:
                if step % args.do_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    train_time = time.time() - tic
                    print (template.format(step, train_loss.result()*len(args.gpu_id), train_time/args.do_log))
                    fine_tuning_time += train_time
                    tic = time.time()

                if step % datasets['train_len'] == 0:

                    tic_ = time.time()
                    test_acc, test_loss = Eval.run(False)

                    tf.summary.scalar('Categorical_loss/train', train_loss.result()*len(args.gpu_id), step=epoch+1)
                    tf.summary.scalar('Categorical_loss/test', test_loss*len(args.gpu_id), step=epoch+1)
                    tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
                    tf.summary.scalar('Accuracy/test', test_acc*100, step=epoch+1)
                    tf.summary.scalar('learning_rate', lr, step=epoch)
                    summary_writer.flush()

                    template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val_loss: {3:0.4f}, val_Acc.: {4:2.2f}'
                    print (template.format(epoch+1, train_loss.result()*len(args.gpu_id), train_accuracy.result()*100,
                                                    test_loss*len(args.gpu_id),  test_acc*100))

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    tic += time.time() - tic_

                    # Save checkpoint
                    if test_acc > best_acc:
                        print('Achieve Best, Saving...')
                        best_acc.assign(test_acc)
                        curr_epoch.assign(epoch + 1)
                        if not os.path.isdir('./checkpoints/'):
                            os.mkdir('./checkpoints/')
                        if not os.path.isdir(ckpt_path):
                            os.mkdir(ckpt_path)

                        manager.save()

                        ckpt.restore(manager.latest_checkpoint)

                        Eval = utils.Evaluation(args, model, strategy, datasets['test'], loss_object)
                        test_acc, test_loss = Eval.run(False)
                        print ('Prediction Accuracy: {:.2f}%'.format(test_acc*100))

        # Restore the weights
        ckpt.restore(manager.latest_checkpoint)

        # Prediction
        Eval = utils.Evaluation(args, model, strategy, datasets['test'], loss_object)
        test_acc, test_loss = Eval.run(False)
        if args.spratio is not None:
            print("print the first 64 element of loaded weight:")
            print(model.layers[2].get_weights()[0][0,0,0,:])
        print ('Prediction Accuracy: {:.2f}%'.format(test_acc*100))
