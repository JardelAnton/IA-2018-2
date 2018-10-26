#task 12 jovani de souza
#intro to tensorflow
#resolver os 3 problemas

#Problema 1: Normalize as características

    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


#Problema 2: Use operações de TensorFlow para criar características, labels, weight, and biases tensors

    features_count = 784
    labels_count = 10

    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))


#Problema 3: Aumente a tava de aprendizado, número de passos e o tamanho do lote para melhor precisão

    batch_size = 128
    epochs = 5
    learning Rate = 0.2
