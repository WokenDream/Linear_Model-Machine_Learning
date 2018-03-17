import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def buildGraph(learning_rate=0.005, useAdam=False):
    # straight from tutorial ex3
    W = tf.Variable(tf.truncated_normal([28*28, 1], stddev=0.5), name='W')
    b = tf.Variable(0.0, name='b')
    X = tf.placeholder(tf.float32, [None, 28*28], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    lamda = tf.placeholder(tf.float32, name='lambda')

    # build linear model/graph
    logits = tf.matmul(X, W) + b
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    mean_entropy = 0.5 * tf.reduce_mean(entropy)
    # entropy_loss = 0.5 * tf.reduce_mean(tf.reduce_mean(entropy, reduction_indices=1, name='squared_error'), name="MSE")
    reg_loss = lamda * tf.nn.l2_loss(W)
    total_loss = mean_entropy + reg_loss
    if useAdam:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss=total_loss)

    return W, b, X, y, lamda, logits, mean_entropy, train



def part_2_1_1(trainData, trainTarget, validData, validTarget, testData, testTarget):
    reshapedTrainData = tf.constant(trainData.reshape(-1, 28 * 28), dtype=tf.float32)
    reshapedValidData = tf.constant(validData.reshape(-1, 28 * 28), dtype=tf.float32)
    reshapedTestData = tf.constant(testData.reshape(-1, 28 * 28), dtype=tf.float32)
    tfValidTarget = tf.constant(validTarget, dtype=tf.float32)
    # tfTestTarget = tf.constant(testTarget, dtype=tf.float32)
    lamda_val = 0.01
    num_iterations = 5000
    num_train = trainData.shape[0]
    batch_size = 500
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)
    i = 0
    for lr in [0.005]: # this is the best after tuning
    # for lr in [0.005, 0.001, 0.0001]:
        i += 1
        W, b, X, y, lamda, logits, mean_entropy, train = buildGraph(lr)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            W_list = []
            train_error_list = []
            train_acc_list = []
            valid_error_list = []
            valid_acc_list = []

            shuffled_inds = np.arange(num_train)
            for epoch in range(num_epochs):
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]

                for j in range(num_batches):
                    batch_trainData = temp_trainData[j * batch_size: (j + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[j * batch_size: (j + 1) * batch_size]
                    _, err, currentW, currentb, currentLogits = sess.run([train, mean_entropy, W, b, logits], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)

                validLogits = tf.matmul(reshapedValidData, currentW) + b
                validEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tfValidTarget, logits=validLogits)
                validErr = 0.5 * sess.run(tf.reduce_mean(validEntropy))
                valid_error_list.append(validErr)

                y_hat = tf.sigmoid(tf.matmul(reshapedTrainData, currentW) + b)
                y_hat_val = sess.run(y_hat)
                y_hat_val[y_hat_val > 0.5] = 1
                y_hat_val[y_hat_val < 0.5] = 0
                acc = np.mean(y_hat_val == trainTarget)
                train_acc_list.append(acc)

                y_hat = tf.sigmoid(tf.matmul(reshapedValidData, currentW) + b)
                y_hat_val = sess.run(y_hat)
                y_hat_val[y_hat_val > 0.5] = 1
                y_hat_val[y_hat_val < 0.5] = 0
                validAcc = np.mean(y_hat_val == validTarget)
                valid_acc_list.append(validAcc)

            # train on the last epoch which is an incomplete epoch
            if (num_iterations_leftover != 0):
                np.random.shuffle(shuffled_inds)
                temp_trainData = trainData[shuffled_inds]
                temp_trainTargets = trainTarget[shuffled_inds]
                for iter in range(num_iterations_leftover):
                    batch_trainData = temp_trainData[iter * batch_size: (iter + 1) * batch_size].reshape(batch_size, -1)
                    batch_targets = temp_trainTargets[iter * batch_size: (iter + 1) * batch_size]
                    _, err, currentW, currentb, currentLogits = sess.run([train, mean_entropy, W, b, logits], feed_dict={
                        X: batch_trainData,
                        y: batch_targets,
                        lamda: lamda_val
                    })
                W_list.append(currentW)
                train_error_list.append(err)

                validLogits = tf.matmul(reshapedValidData, currentW) + b
                validEntropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tfValidTarget, logits=validLogits)
                validErr = 0.5 * sess.run(tf.reduce_mean(validEntropy))
                valid_error_list.append(validErr)

                y_hat = tf.sigmoid(tf.matmul(reshapedTrainData, currentW) + b)
                y_hat_val = sess.run(y_hat)
                y_hat_val[y_hat_val > 0.5] = 1
                y_hat_val[y_hat_val < 0.5] = 0
                acc = np.mean(y_hat_val == trainTarget)
                train_acc_list.append(acc)


                y_hat = tf.sigmoid(tf.matmul(reshapedValidData, currentW) + b)
                y_hat_val = sess.run(y_hat)
                y_hat_val[y_hat_val > 0.5] = 1
                y_hat_val[y_hat_val < 0.5] = 0
                validAcc = np.mean(y_hat_val == validTarget)
                valid_acc_list.append(validAcc)

            # np.save("part2_1_1Wbest.npy", W_list[-1])
            # np.save("part2_1_1W" + str(i), W_list[-1])
            np.save("part2_1_1W" + str("0001"), W_list[-1])
            with open("part_2_1_1.txt", "a") as file:
                file.write("final training err for learning rate = " + str(lr) + ": " + str(err) + "\n")
                file.write("final training acc for learning rate = " + str(lr) + ": " + str(acc) + "\n")
                file.write("final validation err for learning rate = " + str(lr) + ": " + str(validErr) + "\n")
                file.write("final validation acc for learning rate = " + str(lr) + ": " + str(validAcc) + "\n")
            print("final mse for lr =", lr, ":", err)

            plt.subplot(2, 1, 1)
            plt.plot(np.arange(num_epochs + 1), train_error_list)
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(num_epochs + 1), train_acc_list)
            plt.subplot(2, 1, 1)
            plt.plot(np.arange(num_epochs + 1), valid_error_list)
            plt.subplot(2, 1, 2)
            plt.plot(np.arange(num_epochs + 1), valid_acc_list)

    # plt.legend(['learning rate: 0.005', 'learning rate: 0.001', 'learning rate: 0.0001'])
    plt.subplot(2, 1, 1)
    # plt.plot(np.arange(num_epochs + 1), train_error_list)
    plt.title("SGD training - error vs epoch #")
    plt.legend(['best learning rate: 0.001'])
    # plt.legend(['best learning rate: 0.005'], ['best learning rate: 0.001'], ['best learning rate: 0.0001'])
    plt.subplot(2, 1, 2)
    # plt.plot(np.arange(num_epochs + 1), train_acc_list)
    plt.title("SGD training - acc vs epoch #")
    plt.legend(['best learning rate: 0.001'])
    # plt.legend(['best learning rate: 0.005'], ['best learning rate: 0.001'], ['best learning rate: 0.0001'])
    plt.tight_layout()
    plt.savefig("part_2_1_1_train", dpi=600)
    plt.gcf().clear()

    plt.subplot(2, 1, 1)
    # plt.plot(np.arange(num_epochs + 1), valid_error_list)
    # plt.title("SGD validation - error vs epoch #")
    plt.legend(['best learning rate: 0.005'], ['best learning rate: 0.001'], ['best learning rate: 0.0001'])
    plt.subplot(2, 1, 2)
    # plt.plot(np.arange(num_epochs + 1), valid_acc_list)
    plt.title("SGD validation - acc vs epoch #")
    plt.legend(['best learning rate: 0.005'], ['best learning rate: 0.001'], ['best learning rate: 0.0001'])
    # plt.tight_layout()
    plt.savefig("part_2_1_1_valid", dpi=600)
    plt.gcf().clear()

    #load the best W
    # W_best = np.load("part2_1_1Wbest.npy")
    # y_hat = tf.sigmoid(tf.matmul(reshapedTestData, W_best) + 0)
    # with tf.Session() as sess:
    #     y_hat_val = sess.run(y_hat)
    #     y_hat_val[y_hat_val > 0.5] = 1
    #     y_hat_val[y_hat_val < 0.5] = 0
    #     acc = np.mean(y_hat_val == testTarget)
    #     print("best test acc =", acc)
    #     with open("part_2_1_1.txt", "a") as file:
    #         file.write("final test acc = " + str(acc) + "\n")


def part_2_1_2(trainData, trainTarget):
    # reshapedTrainData = tf.constant(trainData.reshape(-1, 28 * 28), dtype=tf.float32)
    lamda_val = 0.01
    lr = 0.001
    batch_size = 500
    num_iterations = 5000
    num_train = trainData.shape[0]
    num_batches = num_train // batch_size
    num_epochs = num_iterations // num_batches
    num_iterations_leftover = num_iterations % num_batches
    print("batch size:", batch_size, "; number of batches", num_batches)
    print("number of epochs:", num_epochs, "; iteration left-overs:", num_iterations_leftover)
    print("number of iterations:", num_iterations)

    W_SGD, b_SGD, X_SGD, y_SGD, lamda_SGD, logits_SGD, mean_entropy_SDG, train_SGD = buildGraph(lr)
    W_Adam, b_Adam, X_Adam, y_Adam, lamda_Adam, logits_Adam, mean_entropy_Adam, train_Adam = buildGraph(lr,True)
    SGD_error_list = []
    SGD_acc_list = []
    Adam_error_list = []
    Adam_acc_list = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        shuffled_inds = np.arange(num_train)
        for epoch in range(num_epochs):
            np.random.shuffle(shuffled_inds)
            temp_trainData = trainData[shuffled_inds]
            temp_trainTargets = trainTarget[shuffled_inds]

            for j in range(num_batches):
                batch_trainData = temp_trainData[j * batch_size: (j + 1) * batch_size].reshape(batch_size, -1)
                batch_targets = temp_trainTargets[j * batch_size: (j + 1) * batch_size]

                # run SGD
                _, err_SGD, currentW_SGD, currentb_SGD, currentLogits_SGD = sess.run(
                    [train_SGD, mean_entropy_SDG, W_SGD, b_SGD, logits_SGD],
                    feed_dict={
                        X_SGD: batch_trainData,
                        y_SGD: batch_targets,
                        lamda_SGD: lamda_val
                    })

                # run Adam
                _, err_Adam, currentW_Adam, currentb_Adam, currentLogits_Adam = sess.run(
                    [train_Adam, mean_entropy_Adam, W_Adam, b_Adam, logits_Adam],
                    feed_dict={
                        X_Adam: batch_trainData,
                        y_Adam: batch_targets,
                        lamda_Adam: lamda_val
                    })
            SGD_error_list.append(err_SGD)
            Adam_error_list.append(err_Adam)

            y_hat = tf.sigmoid(tf.matmul(tf.constant(batch_trainData.reshape(-1, 28 * 28), dtype=tf.float32), currentW_SGD) + b_SGD)
            y_hat_val = sess.run(y_hat)
            y_hat_val[y_hat_val > 0.5] = 1
            y_hat_val[y_hat_val < 0.5] = 0
            acc = np.mean(y_hat_val == trainTarget)
            SGD_acc_list.append(acc)

            y_hat = tf.sigmoid(tf.matmul(reshapedTrainData, currentW_Adam) + b_Adam)
            y_hat_val = sess.run(y_hat)
            y_hat_val[y_hat_val > 0.5] = 1
            y_hat_val[y_hat_val < 0.5] = 0
            acc = np.mean(y_hat_val == trainTarget)
            Adam_acc_list.append(acc)

        # train on the last epoch which is an incomplete epoch
        if (num_iterations_leftover != 0):
            np.random.shuffle(shuffled_inds)
            temp_trainData = trainData[shuffled_inds]
            temp_trainTargets = trainTarget[shuffled_inds]
            for iter in range(num_iterations_leftover):
                batch_trainData = temp_trainData[iter * batch_size: (iter + 1) * batch_size].reshape(batch_size, -1)
                batch_targets = temp_trainTargets[iter * batch_size: (iter + 1) * batch_size]

                _, err_SGD, currentW_SGD, currentb_SGD, currentLogits_SGD = sess.run(
                    [train_SGD, mean_entropy_SDG, W_SGD, b_SGD, logits_SGD],
                    feed_dict={
                        X_SGD: batch_trainData,
                        y_SGD: batch_targets,
                        lamda_SGD: lamda_val
                    })

                # run Adam
                _, err_Adam, currentW_Adam, currentb_Adam, currentLogits_Adam = sess.run(
                    [train_Adam, mean_entropy_Adam, W_Adam, b_Adam, logits_Adam],
                    feed_dict={
                        X_Adam: batch_trainData,
                        y_Adam: batch_targets,
                        lamda_Adam: lamda_val
                    })
            SGD_error_list.append(err_SGD)
            Adam_error_list.append(err_Adam)

            y_hat = tf.sigmoid(tf.matmul(reshapedTrainData, currentW_SGD) + b_SGD)
            y_hat_val = sess.run(y_hat)
            y_hat_val[y_hat_val > 0.5] = 1
            y_hat_val[y_hat_val < 0.5] = 0
            acc = np.mean(y_hat_val == trainTarget)
            SGD_acc_list.append(acc)

            y_hat = tf.sigmoid(tf.matmul(reshapedTrainData, currentW_Adam) + b_Adam)
            y_hat_val = sess.run(y_hat)
            y_hat_val[y_hat_val > 0.5] = 1
            y_hat_val[y_hat_val < 0.5] = 0
            acc = np.mean(y_hat_val == trainTarget)
            Adam_acc_list.append(acc)

        with open("part_2_1_2.txt", "a") as file:
            file.write("SGD final mse: " + str(SGD_error_list[-1]) + "\n")
            file.write("SGD final acc: " + str(SGD_acc_list[-1]) + "\n")
            file.write("Adam final mse: " + str(Adam_error_list[-1]) + "\n")
            file.write("Adam final acc: " + str(Adam_acc_list[-1]) + "\n")

        plt.subplot(2,1,1)
        plt.plot(np.arange(num_epochs + 1), SGD_error_list)
        plt.plot(np.arange(num_epochs + 1), Adam_error_list)
        plt.title("Training error vs epoch #")
        plt.legend(["Plain SGD", "Adam SGD"])

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(num_epochs + 1), SGD_acc_list)
        plt.plot(np.arange(num_epochs + 1), Adam_acc_list)
        plt.title("Training accuracy vs epoch #")
        plt.legend(["Plain SGD", "Adam SGD"])

        plt.tight_layout()
        plt.savefig("part_2_1_2", dpi=600)



def part_2_1():
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
    # part_2_1_1(trainData, trainTarget, validData, validTarget, testData, testTarget)
    part_2_1_2(trainData, trainTarget)


if __name__ == "__main__":
    part_2_1()