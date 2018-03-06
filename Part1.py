import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def buildGraph():
    # straight from tutorial ex3
    W = tf.Variable(tf.truncated_normal([28, 1], stddev=0.5), name='W')
    b = tf.Variable(0.0, name='b')
    X = tf.placeholder(tf.float32, [None, 28], name='X')
    y = tf.placeholder(tf.float32, [None,1], name='y')
    lamda = tf.placeholder(tf.float32, name='lambda')

    # build linear model/graph
    y_hat = tf.matmul(X, W) + b
    mse = tf.reduce_mean(tf.reduce_mean(tf.square(y - y_hat), reduction_indices=1, name='squared_error'), name="MSE")
    reg_loss = lamda * tf.nn.l2_loss(W)
    total_loss = mse + reg_loss

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss=total_loss)

    return W, b, X, y, lamda, y_hat, mse, train


if __name__ == "__main__":
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


    W, b, X, y, lamda, y_hat, mse, train = buildGraph()

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    lamda_val = 0

    w_list = []
    train_error_list = []
    valid_error_list = []
    test_error_list = []

    # 3500/500 = 7 mini-batches
    # 1 iteration = 1 pass through 1 mini-batch
    # 1 epoch = 7 mini-batches (# of mini-batches)
    # => 20000/7 epochs in total
    batch_size = 500
    num_iteration = 20000
    num_train = trainData.shape[0]
    num_batches = num_train / batch_size
    print("number of batches: ", batch_size)

    # TODO: figure out what to do
    for step in range(num_iteration):
        # if this is a new batch
        if step % batch_size == 0:
            batch_inds = np.random.choice(num_train, batch_size, replace=True)
            batch_trainData = trainData[batch_inds]
        _, err, currentW, currentb, yhat = sess.run([train, mse, W, b, y_hat], feed_dict={

        })