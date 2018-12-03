import tensorflow.contrib.slim as slim
import tensorflow as tf

# Variable slim对变量进行了分类：

# Rugular variables 非模型变量（Non-model variables）指的是那些在学习或者评估阶段使用但是在实际的inference中不需要用到的变量。
# 比如说，global_step在学习和评估阶段会用到的变量，但是实际上并不是模型的一部分。类似的，moving average variables也是非模型变量。

weights = slim.variable("weights", shape=[128, 299, 299, 3],
                        initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1),
                        regularizer=slim.l2_regularizer(0.05))

# Model Variables Model variables在学习期间被训练或者fine-tuned，在评估或者推断期间可以从一个checkpoint中加载。
# 模型变量包括使用slim.fully_connected 或者 slim.conv2d创建的变量等。

model_weights = slim.model_variable(name="mode_weight", shape=[10, 299, 299, 3], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1),
                                    regularizer=slim.l2_regularizer(0.004))

# 获取model_variables
model_variables = slim.get_model_variables()

# 获取所有的变量
variables = slim.get_variables()

# Layers

'''
   在原生的Tensorflow中，要定义一些层（比如说卷积层，全连接层，BatchNorm层等）是比较麻烦的。举个例子，神经网络中的卷积层由以下几个步骤组成：
        创建权重和偏置变量
        将输入与权重做卷积运算
        将偏置加到第二步的卷积运算得到的结果中
        使用一个激活函数
        上面的步骤使用原始的Tensorflow代码，实现如下：
            
            input = ...
            with tf.name_scope('conv1_1') as scope:
              kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                       stddev=1e-1), name='weights')
              conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
              biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                   trainable=True, name='biases')
              bias = tf.nn.bias_add(conv, biases)
              conv1 = tf.nn.relu(bias, name=scope)
              
    为了减少重复代码，TF-Slim提供了一些方便高级别更抽象的神经网络层。比如说，卷积层实现如下：
        input=...
        net=slim.conv2d(input,128,[3,3],scope='conv1_1')
'''
# Repeat
'''
slim.repeat

net = ...
net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')

使用TF-Slim中的repeat操作：
    net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
    net=slim.max_pool2d(net,[2,2],scope='pool2')
    
 上面例子中，slim.repeat会自动给每一个卷积层的scopes命名为'conv3/conv3_1', 'conv3/conv3_2' 和 'conv3/conv3_3'
'''
# Stack
'''
TF-Slim的 slim.stack操作允许用户用不同的参数重复调用同一种操作。
slim.stack也为每一个被创建的操作创建一个新的tf.variable_scope。
比如说，下面是一种简单的方式来创建多层感知器（Multi-Layer Perceptron (MLP)）：

# Verbose way:详细的方式 Verbose 冗长的，啰嗦的
x = slim.fully_connected(x, 32, scope='fc/fc_1')
x = slim.fully_connected(x, 64, scope='fc/fc_2')
x = slim.fully_connected(x, 128, scope='fc/fc_3')

# Equivalent, TF-Slim way using slim.stack:
slim.stack(x,slim.fully_connected,[32,64,128],scope='fc')

在上面的例子中，slim.stack调用了slim.fully_connected三次。
类似的，我们可以使用stack来简化多层的卷积层。

# Verbose way:
x = slim.conv2d(x, 32, [3, 3], scope='core/core_1')
x = slim.conv2d(x, 32, [1, 1], scope='core/core_2')
x = slim.conv2d(x, 64, [3, 3], scope='core/core_3')
x = slim.conv2d(x, 64, [1, 1], scope='core/core_4')

#Using stack
slim.stack(s,slim.conv2d,[(32,[3,3]),(32,[1,1]),(64,[3,3]),(64,[1,1])],scope='core')


Scopes
    除了Tensorflow中作用域（scope）之外（name_scope, variable_scope），TF-Slim增加了新的作用域机制，称为arg_scope。
    这个新的作用域允许使用者明确一个或者多个操作和一些参数，这些定义好的操作或者参数会传递给arg_scope内部的每一个操作。下面举例说明。
    先看如下代码片段：
    
    net = slim.conv2d(inputs, 64, [11, 11], 4, padding='SAME',
                  
    net = slim.conv2d(net, 128, [11, 11], padding='VALID',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
    net = slim.conv2d(net, 256, [11, 11], padding='SAME',
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')
    从上面的代码中可以清楚的看出来，有3层卷积层，其中很多超参数都是一样的。两个卷积层有相同的padding，所weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                  weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')有三个卷积层有相同的weights_initializer和weight_regularizer。
    上面的代码包含了大量重复的值，其中一种解决方法是使用变量来说明一些默认的值：                
    padding = 'SAME'
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    regularizer = slim.l2_regularizer(0.0005)
    net = slim.conv2d(inputs, 64, [11, 11], 4,
                      padding=padding,
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv1')
    net = slim.conv2d(net, 128, [11, 11],
                      padding='VALID',
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv2')
    net = slim.conv2d(net, 256, [11, 11],
                      padding=padding,
                      weights_initializer=initializer,
                      weights_regularizer=regularizer,
                      scope='conv3')
    上面的解决方案其实并没有减少代码的混乱程度。通过使用arg_scope，我们可以既可以保证每一层使用相同的值，也可以简化代码：
    inputs=...
    with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net=slim.conv2d(inputs,64,[11.11],scope="conv1")
        net=slim.conv2d(net,128,[11,11],padding='VALID',scope='conv2')
        net=slim.conv2d(net,256,[11,11],scope='conv3')
        
    上面的例子表明，使用arg_scope可以使得代码变得更整洁、更干净并且更加容易维护。
    注意到，在arg_scope中规定的参数值，它们可以被局部覆盖。比如说，上面的padding参数被设置成‘SAME’，
    但是在第二个卷积层中用‘VALID’覆盖了这个参数。
    
    我们也可以嵌套使用arg_scope，在相同的作用域内使用多个操作。举例如下：
    inputs = ...
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.conv2d(net, 256, [5, 5], weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                              scope='conv2')
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc')
    在上面的例子中，在第一个arg_scope中，卷积层和全连接层被应用于相同的权重初始化和权重正则化；在第二个arg_scope中，额外的参数仅仅对卷积层conv2d起作用。
    
    # VGG16 slim版
    def vgg16(inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope="pool1")
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope="pool3")
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope="pool4")
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope="pool5")
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope="dropout6")
            net=slim.fully_connected(net,4096,scope="fc7")
            net=slim.dropout(net,0.5,scope="dropout7")
            net=slim.fully_connected(net,1000,activation_fn=None,scope="fc8")
            return net
    
    Training Models
       训练Tensorflow模型要求一个模型、一个loss function、梯度计算和一个训练的程序，
       用来迭代的根据loss计算模型权重的梯度和更新权重。TF-Slim提供了loss function和一些帮助函数，来运行训练和评估。
    Losses
       Loss function定义了一个我们需要最小化的量。对于分类问题，主要是计算真正的分布与预测的概率分布之间的交叉熵。对于回归问题，主要是计算预测值与真实值均方误差。
       特定的模型，比如说多任务学习模型，要求同时使用多个loss function；换句话说，最终被最小化的loss function是多个其他的loss function之和。
       比如说，一个同时预测图像中场景的类型和深度的模型，该模型的loss function就是分类loss和深度预测loss之和（the sum of the classification loss and depth prediction loss）。
       TF-Slim通过losses模块为用户提供了一种机制，使得定义loss function变得简单。比如说，下面的是我们想要训练VGG网络的简单示例：
       import tensorflow as tf
        import tensorflow.contrib.slim.nets as nets
        vgg = nets.vgg
        
        # Load the images and labels.
        images, labels = ...
        
        # Create the model.
        predictions, _ = vgg.vgg_16(images)
        
        # Define the loss functions and get the total loss.
        loss = slim.losses.softmax_cross_entropy(predictions, labels)
        
    在上面这个例子中，我们首先创建一个模型（利用TF-Slim的VGG实现），然后增加了标准的分类loss。
    现在，让我们看看当我们有一个多个输出的多任务模型的情况：
    
    # Load the images and labels.
    images, scene_labels, depth_labels = ...
    
    # Create the model.
    scene_predictions, depth_predictions = CreateMultiTaskModel(images)
    
    # Define the loss functions and get the total loss.
    classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
    sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
    
    # The following two lines have the same effect:
    total_loss = classification_loss + sum_of_squares_loss
    total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
   在这个例子中，我们有2个loss，是通过调用slim.losses.softmax_cross_entropy 和 slim.losses.sum_of_squares得到。
   我们可以将这两个loss加在一起或者调用slim.losses.get_total_loss()来得到全部的loss（total_loss）。这是如何工作的？
   当你通过TF-Slim创建一个loss时，TF-Slim将loss加到一个特殊的TensorFlow collection of loss functions。这使得你既可以手动得管理全部的loss，也可以让TF-Slim来替你管理它们。
   
   如果你想让TF-Slim为你管理losses但是你有一个自己实现的loss该怎么办？loss_ops.py 也有一个函数可以将你自己实现的loss加到 TF-Slims collection中。举例如下： 
   
   # Load the images and labels.
    images, scene_labels, depth_labels, pose_labels = ...
    
    # Create the model.
    scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)
    
    # Define the loss functions and get the total loss.
    classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
    sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
    pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
    slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.
    
    # The following two ways to compute the total loss are equivalent:
    regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
    total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss
    
    # (Regularization Loss is included in the total loss by default).
    total_loss2 = slim.losses.get_total_loss() 
    
    在这个例子中，我们既可以手动的计算的出全部的loss function，也可以让TF-Slim知道这个额外的loss然后让TF-Slim处理这个loss。
    
    
    Training Loop
       TF-Slim提供了一个简单但是很强的用于训练模型的工具（在 learning.py）。其中包括一个可以重复测量loss，计算梯度和将模型保存到磁盘的训练函数。举个例子，一旦我们定义好了模型，
       loss function和最优化方法，我们可以调用slim.learning.create_train_op 和 slim.learning.train来实现优化。
       
        g = tf.Graph()
        # Create the model and specify the losses...
        ...
        
        total_loss = slim.losses.get_total_loss()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        
        # create_train_op ensures that each time we ask for the loss, the update_ops
        # are run and the gradients being computed are applied too.
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        logdir = ... # Where checkpoints are stored.
        
        slim.learning.train(
            train_op,
            logdir,
            number_of_steps=1000,
            save_summaries_secs=300,
            save_interval_secs=600):
            
        在这个例子中，提供给slim.learning.train的参数有1）train_op，用于计算loss和梯度，2）logdir用于声明checkpoints和event文件保存的路径。
        我们可以用number_of_steps参数来限制梯度下降的步数；
        save_summaries_secs=300表明我们每5分钟计算一次
        summaries，save_interval_secs=600表明我们每10分钟保存一次模型的checkpoint。
        
        Working Example: Training the VGG16 Model
        下面是训练一个VGG网络的例子。

        import tensorflow as tf
        import tensorflow.contrib.slim.nets as nets
        
        slim = tf.contrib.slim
        vgg = nets.vgg
        
        ...
        
        train_log_dir = ...
        if not tf.gfile.Exists(train_log_dir):
          tf.gfile.MakeDirs(train_log_dir)
        
        with tf.Graph().as_default():
          # Set up the data loading:
          images, labels = ...
        
          # Define the model:
          predictions = vgg.vgg_16(images, is_training=True)
        
          # Specify the loss function:
          slim.losses.softmax_cross_entropy(predictions, labels)
        
          total_loss = slim.losses.get_total_loss()
          tf.summary.scalar('losses/total_loss', total_loss)
        
          # Specify the optimization scheme:
          optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
        
          # create_train_op that ensures that when we evaluate it to get the loss,
          # the update_ops are done and the gradient updates are computed.
          train_tensor = slim.learning.create_train_op(total_loss, optimizer)
        
          # Actually runs training.
          slim.learning.train(train_tensor, train_log_dir)

    Fine-Tuning Existing Models
    Brief Recap on Restoring Variables from a Checkpoint
       当一个模型被训练完毕之后，它可以从一个给定的checkpoint中使用tf.train.Saver()来恢复变量。
       在很多情况下，tf.train.Saver()提供一个简答的机制来恢复所有变量或者一部分变量
       
       # Create some variables.
        v1 = tf.Variable(..., name="v1")
        v2 = tf.Variable(..., name="v2")
        ...
        # Add ops to restore all the variables.
        restorer = tf.train.Saver()
        
        # Add ops to restore some variables.
        restorer = tf.train.Saver([v1, v2])
        
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        with tf.Session() as sess:
          # Restore variables from disk.
          restorer.restore(sess, "/tmp/model.ckpt")
          print("Model restored.")
          # Do some work with the model
          ...
    
    Partially Restoring Models
       在一个新的数据集或者一个新的任务上fine-tune一个预训练的模型通常是比较受欢迎的。我们可以使用TF-Slim的helper函数来选择想要恢复的一部分变量：
        # Create some variables.
        v1 = slim.variable(name="v1", ...)
        v2 = slim.variable(name="nested/v2", ...)
        ...
        
        # Get list of variables to restore (which contains only 'v2'). These are all
        # equivalent methods:
        variables_to_restore = slim.get_variables_by_name("v2")
        # or
        variables_to_restore = slim.get_variables_by_suffix("2")
        # or
        variables_to_restore = slim.get_variables(scope="nested")
        # or
        variables_to_restore = slim.get_variables_to_restore(include=["nested"])
        # or
        variables_to_restore = slim.get_variables_to_restore(exclude=["v1"])
        
        # Create the saver which will be used to restore the variables.
        restorer = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
          # Restore variables from disk.
          restorer.restore(sess, "/tmp/model.ckpt")
          print("Model restored.")
          # Do some work with the model
          ...
          
    Fine-Tuning a Model on a different task
       考虑这么一种情况：我们有一个预训练好的VGG16模型，该模型是在ImageNet数据集上训练好的，有1000类。
       然而，我们想要将其应用到只有20类的Pascal VOC数据集上。为了实现这个，我们可以使用不包括最后一层的预训练模型来初始化我们的新模型。
       
        # Load the Pascal VOC data
        image, label = MyPascalVocDataLoader(...)
        images, labels = tf.train.batch([image, label], batch_size=32)
        
        # Create the model
        predictions = vgg.vgg_16(images)
        
        train_op = slim.learning.create_train_op(...)
        
        # Specify where the Model, trained on ImageNet, was saved.
        model_path = '/path/to/pre_trained_on_imagenet.checkpoint'
        
        # Specify where the new model will live:
        log_dir = '/path/to/my_pascal_model_dir/'
        
        # Restore only the convolutional layers:
        variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
        init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)
        
        # Start training.
        slim.learning.train(train_op, log_dir, init_fn=init_fn)
        
    Evaluating Models.
       一旦我们已经训练好了一个模型（或者模型正在训练之中），我们想要看看模型的实际表现能力。这个可以通过使用一些评估度量来实现，该度量可以对模型的表现能力评分。
       而评估代码实际上是加载数据，做预测，将预测结果与真实值做比较，最后得到得分。这个步骤可以运行一次或者周期重复。
 
    Metrics
       我们将度量定义为一个性能度量，它不是一个loss函数（losses是在训练的时候直接最优化），但我们仍然感兴趣的是评估模型的目的。比如说，我们想要最优化log loss，
       但是我们感兴趣的度量可能是F1得分（test accuracy），或者是Intersection Over Union score（这是不可微的，因此不能作为损失使用）。
       TF-Slim提供了一些使得评估模型变得简单的度量操作。计算度量的值可以分为以下三个步骤：
        初始化（Initialization）：初始化用于计算度量的变量
        聚合（Aggregation）：使用操作（比如求和操作）来计算度量
        终止化（Finalization）：（可选的）使用最终的操作来计算度量值，比如说计算均值，最小值，最大值等。
         
        举个例子，为了计算mean_absolute_error，2个变量，count 和 total变量被初始化为0。在聚合期间，我们观测到一些预测值和标签值，
        计算它们的绝对差值然后加到total中。每一次我们观测到新的一个数据，我们增加count。最后，在Finalization期间，total除以count来获得均值mean。
    下面的示例演示了声明度量标准的API。由于度量经常在测试集上进行评估，因此我们假设使用的是测试集。
        images, labels = LoadTestData(...)
        predictions = MyModel(images)
        
        mae_value_op, mae_update_op = slim.metrics.streaming_mean_absolute_error(predictions, labels)
        mre_value_op, mre_update_op = slim.metrics.streaming_mean_relative_error(predictions, labels)
        pl_value_op, pl_update_op = slim.metrics.percentage_less(mean_relative_errors, 0.3)       
     如示例所示，一个度量的创建返回两个值：value_op和update_op。value_op是一个幂等操作，它返回度量的当前值。update_op是执行上面提到的聚合步骤的操作，以及返回度量的值。
       跟踪每个value_op和update_op是很费力的。为了解决这个问题，TF-Slim提供了两个便利功能：
       # Aggregates the value and update ops in two lists:
        value_ops, update_ops = slim.metrics.aggregate_metrics(
            slim.metrics.streaming_mean_absolute_error(predictions, labels),
            slim.metrics.streaming_mean_squared_error(predictions, labels))
        
        # Aggregates the value and update ops in two dictionaries:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
            "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
            
    Working example: Tracking Multiple Metrics
 
       将代码全部放在一起：
       
    import tensorflow as tf
    import tensorflow.contrib.slim.nets as nets
    
    slim = tf.contrib.slim
    vgg = nets.vgg
    
    
    # Load the data
    images, labels = load_data(...)
    
    # Define the network
    predictions = vgg.vgg_16(images)
    
    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
        "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
    })
    
    # Evaluate the model using 1000 batches of data:
    num_batches = 1000
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
    
      for batch_id in range(num_batches):
        sess.run(names_to_updates.values())
    
      metric_values = sess.run(names_to_values.values())
      for metric, value in zip(names_to_values.keys(), metric_values):
        print('Metric %s has value: %f' % (metric, value))
    }) 
    
    
Evaluation Loop
       TF-Slim提供了一个评估模块(evaluation.py)，它包含了使用来自 metric_ops.py 模块编写模型评估脚本的辅助函数。
       这些功能包括定期运行评估、对数据批量进行评估、打印和汇总度量结果的功能。
       import tensorflow as tf

        slim = tf.contrib.slim
        
        # Load the data
        images, labels = load_data(...)
        
        # Define the network
        predictions = MyModel(images)
        
        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.accuracy(predictions, labels),
            'precision': slim.metrics.precision(predictions, labels),
            'recall': slim.metrics.recall(mean_relative_errors, 0.3),
        })
        
        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
          op = tf.summary.scalar(metric_name, metric_value)
          op = tf.Print(op, [metric_value], metric_name)
          summary_ops.append(op)
        
        num_examples = 10000
        batch_size = 32
        num_batches = math.ceil(num_examples / float(batch_size))
        
        # Setup the global step.
        slim.get_or_create_global_step()
        
        output_dir = ... # Where the summaries are stored.
        eval_interval_secs = ... # How often to run the evaluation.
        slim.evaluation.evaluation_loop(
            'local',
            checkpoint_dir,
            log_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=eval_interval_secs)
'''

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(weights).shape)
    print(sess.run(model_weights).shape)
    print(sess.run(model_weights))
    print((sess.run(model_variables)))
    print(sess.run(variables).__len__())
    pass
