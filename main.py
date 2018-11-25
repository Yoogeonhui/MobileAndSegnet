import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from model import Model
from dataload import DataLoader
import image_convert
import tensorflow.contrib.slim as slim
from matplotlib import colors

options = {
    'batch_input':'F:\synthia\RGB',
    'batch_GT': 'F:\synthia\GT',
    'optimizer':'Adam',
    'learningRate': 5e-5,
    'checkPointDirectory':'./model/',
    'checkPointName':'mobileSegNet.ckpt',
    'summaryTrainDirectory': './summaryTrain',
    'batchSize': 5,
    'inputWidth': 16*16,
    'inputHeight': 16*9,
    'summaryBatch': 15,
    'saveBatch': 30,
    'labelNumber': len(image_convert.label_to_pixel_dict)
}

# 이거 순서 R, G, B 임.


if __name__ == '__main__':
    model = Model(options['inputWidth'], options['inputHeight'], options['labelNumber'])
    loss = model.get_loss()
    saver = tf.train.Saver()

    epoch_update = tf.placeholder(tf.int32, shape=())
    batch_update = tf.placeholder(tf.int32, shape=())

    epoch_tensor = tf.Variable(0, trainable=False, dtype= tf.int32)
    batch_tensor = tf.Variable(0, trainable=False, dtype= tf.int32)

    assign_epoch = tf.assign(epoch_tensor, epoch_update)
    assign_batch = tf.assign(batch_tensor, batch_update)
    optimizer = tf.train.AdamOptimizer(learning_rate = options['learningRate'])
    if options['optimizer']=='SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = options['learningRate'])

    train = optimizer.minimize(loss)

    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    tf.summary.scalar('loss', loss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(options['summaryTrainDirectory'], sess.graph)
        sess.run(tf.global_variables_initializer())

        if tf.train.get_checkpoint_state(options['checkPointDirectory']) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(options['checkPointDirectory']))
            print('Loaded')

        now_epoch, now_batch = sess.run([epoch_tensor, batch_tensor])

        batch_loader = DataLoader('F:\synthia\RGB\\', 'F:\synthia\GT\\', start_batch=now_batch, batch_size=options['batchSize'])
        fig = plt.figure(figsize = (20,20))
        cmap = colors.ListedColormap(['#000000', '#808080', '#800000', '#804080', '#0000C0', '#404080','#808000', '#C0C080', '#400080','#C08080','#404000', '#0080C0'])
        ax_rgb = fig.add_subplot(2,2,1)
        ax_ground = fig.add_subplot(2,2,2)
        ax_predict = fig.add_subplot(2,2,3)
        plt.ion()
        plt.show()
        epoch = now_epoch
        while epoch<100:
            # 맨 앞에 한번 더 돌리지 뭐
            while True:
                next_epoch, input_array, label_array = batch_loader.get_next_batch()
                train_dict = {model.input_tensor: input_array, model.label_tensor: label_array}
                _, labels, now_loss, summary = sess.run([train, model.label_result, loss, merged], feed_dict=train_dict)
                print('loss', now_loss,' in epoch: ',epoch,'th batch num: ', batch_loader.batch_num)
                if batch_loader.batch_num % options['summaryBatch'] == 0:
                    ax_rgb.clear()
                    ax_rgb.imshow(image_convert.convert_bgr_to_rgb(input_array[0]))
                    ax_ground.clear()
                    ax_ground.imshow(label_array[0], cmap= cmap)
                    ax_predict.clear()
                    ax_predict.imshow(labels[0], cmap= cmap)
                    plt.draw()
                    plt.pause(0.01)
                    train_writer.add_summary(summary, batch_loader.batch_num)

                if batch_loader.batch_num % options['saveBatch'] == 0:
                    sess.run([assign_batch, assign_epoch], feed_dict={epoch_update: epoch, batch_update: batch_loader.batch_num})
                    saver.save(sess, options['checkPointDirectory']+options['checkPointName'])

                if next_epoch:
                    break
            epoch += 1
