import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


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


def linear_regression(lam, batch_size, learning_rate, num_batch, n_iter):
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]


    x = tf.placeholder("float", shape=[batch_size, 784])
    y = tf.placeholder("float")

    #initial guess
    w = tf.Variable(tf.truncated_normal([784, 1], mean=0.0, stddev=0.5, dtype=tf.float32), name="w")
    b = tf.Variable(tf.zeros([1,batch_size]), name="bias")

    y_model = tf.matmul(x, w) + b

    mse = 0.5 * tf.reduce_mean(tf.reduce_mean(tf.square(y - y_model), reduction_indices=1))
    loss_function = lam * tf.nn.l2_loss(w)
    error = mse + loss_function

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(error)

    model = tf.global_variables_initializer()

    errors = []
    w_v = []
    with tf.Session() as session:
        session.run(model)

        shuffled_ind = np.arange(trainData.shape[0])
        for j in range(n_iter // num_batch):
            np.random.shuffle(shuffled_ind)
            temp_trainData = trainData[shuffled_ind]
            temp_trainTarget = trainTarget[shuffled_ind]

            for i in range(num_batch):
                x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
                y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
                # x_value = x_batches[i]
                # x_value = np.reshape(x_value, [784, batch_size])
                # y_value = y_batches[i]
                _, error_value, w_value = session.run([train_op, error, w], feed_dict={x: x_value, y: y_value})
                errors.append(error_value)
                w_v.append(w_value)

        for k in range(n_iter % num_batch):
            x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
            y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
            _, error_value, w_value = session.run([train_op, error, w], feed_dict={x: x_value, y: y_value})
            errors.append(error_value)
            w_v.append(w_value)
        return w_v, errors


def logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter):
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]


    x = tf.placeholder("float", shape=[batch_size, 784])
    y = tf.placeholder("float")

    #initial guess
    w = tf.Variable(tf.truncated_normal([784, 1], mean=0.0, stddev=0.5, dtype=tf.float32), name="w")
    b = tf.Variable(tf.zeros([1,batch_size]), name="bias")
    logit = tf.matmul(x, w) + b

    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.transpose(y), logits=logit)
    loss_function = lam / 2 * tf.square(tf.norm(w))
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    error = cross_entropy_loss + loss_function

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=error)

    model = tf.global_variables_initializer()

    errors = []
    w_v = []
    with tf.Session() as session:
        session.run(model)

        shuffled_ind = np.arange(trainData.shape[0])
        for j in range(n_iter//num_batch):
            np.random.shuffle(shuffled_ind)
            temp_trainData = trainData[shuffled_ind]
            temp_trainTarget = trainTarget[shuffled_ind]

            for i in range(num_batch):
                x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
                y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
                _, error_value, w_value = session.run([train_op, error, w], feed_dict={x: x_value, y: y_value})
                errors.append(error_value)
                w_v.append(w_value)

        for k in range(n_iter%num_batch):
            x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
            y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
            _, error_value, w_value = session.run([train_op, error, w], feed_dict={x: x_value, y: y_value})
            errors.append(error_value)
            w_v.append(w_value)

        return w_v, errors

def get_accuracy(w, trainData, trainTarget, size):
    x_value = trainData
    x_value = np.reshape(x_value, [size, 784])
    y_value = trainTarget
    y_model = np.matmul(x_value, w)
    labels = y_model
    labels[labels >= 0.5] = 1
    labels[labels < 0.5] = 0
    accuracy = np.mean(labels == y_value)
    return accuracy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_accuracy_sig(w, trainData, trainTarget, size):

    with tf.Session():
        x_value = trainData
        x_value = np.reshape(x_value, [size, 784])
        y_value = trainTarget
        y_model = np.matmul(x_value, w)
        sigmoid_v = np.vectorize(sigmoid)

        labels = sigmoid_v(y_model)
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        accuracy = np.mean(labels == y_value)
        return accuracy

def main():
    lam = 0
    batch_size = 500 #if you change this
    learning_rate = 0.001
    num_batch = 7 #change this
    n_iter = 5000

    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

    accuracy_linear = []
    accuracy_logistic = []
    w, errors1 = (linear_regression(lam, batch_size, learning_rate, num_batch, n_iter))
    w2, errors = logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter)


    # plt.xlabel("epoch")
    # plt.ylabel("loss function")
    # to_plot1 = errors1[0:n_iter - 1:num_batch]
    # plt.plot(to_plot1, label='linear regression')
    # plt.plot(errors, label='logistic regression')


    #accuracy
    w = w[0:n_iter - 1:num_batch]
    for i in w:
        accuracy_linear.append(get_accuracy(i, trainData, trainTarget, 3500))

    w2 = w2[0:n_iter - 1:num_batch]
    for j in w2:
        accuracy_logistic.append(get_accuracy_sig(j, trainData, trainTarget, 3500))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(accuracy_linear, label='linear regression')
    plt.plot(accuracy_logistic, label='logistic regression')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()