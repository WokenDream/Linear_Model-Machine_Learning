import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def buildGraph(learning_rate=0.005):
    # straight from tutorial ex3
    W = tf.Variable(tf.truncated_normal([28*28, 1], stddev=0.5), name='W')
    b = tf.Variable(0.0, name='b')
    X = tf.placeholder(tf.float32, [None, 28*28], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    lamda = tf.placeholder(tf.float32, name='lambda')

    # build linear model/graph
    y_hat = tf.matmul(X, W) + b
    mse = tf.reduce_mean(tf.reduce_mean(tf.square(y - y_hat), reduction_indices=1, name='squared_error'), name="MSE")
    reg_loss = lamda * tf.nn.l2_loss(W)
    total_loss = mse + reg_loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss=total_loss)

    return W, b, X, y, lamda, y_hat, mse, train

def part_1_1(trainData, trainTarget):

    lamda_val = 0
    batch_size = 500
    num_iterations = 20000
    num_train = trainData.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    for lr in [0.005, 0.001, 0.0001]:
        W, b, X, y, lamda, y_hat, mse, train = buildGraph(lr)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            W_list = []
            train_error_list = []

            shuffled_inds = np.arange(num_train)
            # follow lecture notes
            for epoch in range(num_epochs):
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]

                for j in range(num_batches):
                    batch_trainData = temp_trainData[j * batch_size: (j + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[j * batch_size: (j + 1) * batch_size]
                    _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)
                # print("epoch ", epoch, " train error: ", err)

            # train on the last epoch which is an incomplete epoch
            if (num_iterations_leftover != 0):
                print("# of batches does not divide # of iterations, training over the remaining")
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]
                for iter in range(num_iterations_leftover):
                    batch_trainData = temp_trainData[iter * batch_size: (iter + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[iter * batch_size: (iter + 1) * batch_size]
                    _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)

            with open("part_1_1.txt", "a") as file:
                file.write("final mse for learning rate = " + str(err) + "\n")
            print("final mse:", err)
            # print("epoch ", epoch, " train error: ", err)

            plt.plot(np.arange(num_epochs + 1), train_error_list)
    plt.title("SGD training - error vs epoch #")
    plt.legend(['learning rate: 0.005', 'learning rate: 0.001', 'learning rate: 0.0001'])
    plt.savefig("part_1_1")
    plt.show()

def part_1_2(trainData, trainTarget):
    lr = 0.005
    lamda_val = 0
    num_iterations = 20000
    num_train = trainData.shape[0]
    for batch_size in [500, 1500, 3500]:

        num_batches = num_train // batch_size
        num_epochs = num_iterations // num_batches
        num_iterations_leftover = num_iterations % num_batches
        print("batch size:", batch_size, "; number of batches", num_batches)
        print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
        print("number of iterations:", num_iterations)

        W, b, X, y, lamda, y_hat, mse, train = buildGraph(lr)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            W_list = []
            train_error_list = []

            shuffled_inds = np.arange(num_train)
            start_time_sec = time.time()
            for epoch in range(num_epochs):
                # print("epoch ", epoch)
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]

                for j in range(num_batches):
                    batch_trainData = temp_trainData[j * batch_size: (j + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[j * batch_size: (j + 1) * batch_size]
                    _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })

                if (num_batches * batch_size < num_train):
                    # print("batch size does not divide # of training examples, training over the remaining")
                    batch_trainData = temp_trainData[num_batches * batch_size:].reshape(num_train - num_batches * batch_size, -1)
                    batch_targets = temp_trainTargets[num_batches * batch_size:]
                    _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)

            if (num_iterations_leftover != 0):
                # print("# of batches does not divide # of iterations, training over the remaining")
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]
                for iter in range(num_iterations_leftover):
                    batch_trainData = temp_trainData[iter * batch_size: (iter + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[iter * batch_size: (iter + 1) * batch_size]
                    _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)

            stop_time_sec = time.time()
            with open("part_1_2.txt", "a") as file:
                file.write("for batch size = " + str(batch_size) + ":\n")
                file.write("\tmse = " + str(err) + "; time taken = " + str(stop_time_sec - start_time_sec) + "\n")
            print("final mse:", err)
            print("time taken: ", stop_time_sec - start_time_sec)
            if (num_iterations_leftover != 0):
                plt.plot(np.arange(num_epochs + 1), train_error_list, linewidth=0.8)
            else:
                plt.plot(np.arange(num_epochs), train_error_list, linewidth=0.8)

    plt.title("SGD training - error vs epoch #")
    plt.legend(['batch size: 500', 'batch size: 1500', 'batch size: 3500'])
    plt.savefig("part_1_2", dpi=600)
    plt.show()


if __name__ == "__main__":
    # plt.plot(np.arange(2500), np.arange(2500), linewidth=1.0)
    # plt.show()
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    print("trainData shape", trainData.shape)
    print("trainTarget shape", trainTarget.shape)
    print("validData shape", validData.shape)
    print("validTarget shape", validTarget.shape)
    print("testData shape", testData.shape)
    print("testTarget shape", testTarget.shape)
    # part_1_1(trainData, trainTarget)
    part_1_2(trainData, trainTarget)