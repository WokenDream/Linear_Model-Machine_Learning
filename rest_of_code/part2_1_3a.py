import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def logistic_regression_gradient(lam, batch_size, learning_rate, num_batch, n_iter):
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

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    model = tf.global_variables_initializer()

    errors = []
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
                # x_value = x_batches[i]
                # x_value = np.reshape(x_value, [784, batch_size])
                # y_value = y_batches[i]
                _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
                errors.append(error_value)

        for k in range(n_iter%num_batch):
            x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
            y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
            _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
            errors.append(error_value)
        w_value = session.run(w)
        return w_value, errors


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
                # x_value = x_batches[i]
                # x_value = np.reshape(x_value, [784, batch_size])
                # y_value = y_batches[i]
                _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
                errors.append(error_value)

        for k in range(n_iter%num_batch):
            x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
            y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
            _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
            errors.append(error_value)
        w_value = session.run(w)
        return w_value, errors


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def validation_accuracy(w):
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

    sigmoid_v = np.vectorize(sigmoid)
    x_value = validData
    x_value = np.reshape(x_value, [100, 784])
    y_value = validTarget
    y_model = np.matmul(x_value, w)
    labels = sigmoid_v(y_model)
    labels[labels>=0.5]=1
    labels[labels<0.5]=0
    accuracy = np.mean(labels==y_value)
    return accuracy


def main():
    lam = 0
    batch_size = 500 #if you change this
    learning_rate = 0.025
    num_batch = 7 #change this
    n_iter = 5000

    errors1 = []
    w, errors1 = (logistic_regression_gradient(lam, batch_size, learning_rate, num_batch, n_iter))
    w, errors2 = logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter)
    #accuracy = validation_accuracy(w)
    #print(accuracy, " with learning rate", learning_rate)
    plt.xlabel("epoch")
    plt.ylabel("loss function")
    to_plot1 = errors1[0:n_iter - 1:num_batch]
    to_plot2 = errors2[0:n_iter - 1:num_batch]
    plt.plot(to_plot1, label='gradient descent')
    plt.plot(to_plot2, label='adam')
    plt.legend()

    plt.show()
    #print("final MSE is: " + str(errors[n_iter - 1]))


if __name__ == "__main__":
    main()
