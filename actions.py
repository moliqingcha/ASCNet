import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
from ascnet import Ascnet
from deeplabv3 import Deeplabv3
from deeplabv3plus import Deeplabv3plus
from unet import Unet
from pspnet import Pspnet
from segnet import Segnet
import ops

"""
Actions类功能：包括最重要的三种操作函数train(),test(),predcit()
---------------
| 各函数功能：|
---------------
-------------------------------- 网络配置 --------------------------------------
函数configure_networks_single()：使用单个GPU时的网络配置
函数configure_networks_multi()：使用多个GPU时的网络配置，需要调用函数average_gradients()
函数average_gradients()：计算多GPU时的平均梯度
-------------------------------- 训练、验证、测试 --------------------------------------
函数train()：训练
函数test()：验证
函数predict()：测试
-------------------------------- 保存 --------------------------------------
函数config_summary()：用于配置summary
函数save_summary()：用于保存summary
函数save()：用于保存模型
函数reload()：用于加载模型
---------------
|  注意事项： |
---------------
1. 

"""
class Actions(object):
#———————————————————————————— 请在这里设置参数 @_@ —————————————————————————#
    def __init__(self, sess, conf):
        
        #——————————————  step：1  ——————————————#——设置class内会用到的参数———#
        # 1）传递main中定义的基本参数
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        # 2）设置一些需要的参数
        self.data_format = 'NHWC'
        self.axis, self.channel_axis, self.batch_axis = (1, 2), 3, 0
        self.input_shape = [conf.batchsize, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batchsize, conf.height, conf.width]
        
        #——————————————  常用参数，只看这里就行  ——————————————#
        
        
        
        #——————————————  step：2  ——————————————#——设置一些保存模型需要的文件夹———#
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
            
        #——————————————  step：3  ——————————————#——配置网络，单个GPU，多个GPU分别设置———#
        if self.conf.gpu_num==1:
            self.configure_networks_single()
        else:
            self.configure_networks_multi()
#———————————————————————————— configure_networks_single —————————————————————————# 
    """
    函数功能：当使用单个GPU时的网络配置
    输入：无
    输出：无
    """
    def configure_networks_single(self):
        
        #——————————————  step：1  ——————————————#
        # 设置X\Y的容器；把标签转换成one-hot形式，为了下一步计算loss时使用；
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(tf.int64, self.output_shape, name='annotations')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        expand_annotations = tf.expand_dims(self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(expand_annotations, axis=[self.channel_axis],name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        
        #——————————————  step：2  ——————————————#
        # 根据搭建的模型计算预测出来的Y；除了预测值可能还包括一些其他想输出的参数；
        if self.conf.network_name=="ascnet":
            model = Ascnet(self.sess, self.conf, self.is_train)
            self.predictions, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="segnet":
            model = Segnet(self.sess, self.conf, self.is_train)
            self.predictions, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="deeplabv3":
            model = Deeplabv3(self.sess, self.conf, self.is_train)
            self.predictions, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="deeplabv3plus":
            model = Deeplabv3plus(self.sess, self.conf, self.is_train)
            self.predictions, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="unet":
            model = Unet(self.sess, self.conf, self.is_train)
            self.predictions, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="pspnet":
            model = Pspnet(self.sess, self.conf, self.is_train)
            self.predictions, self.rates, self.pred2 = model.inference(self.inputs)
       
        #——————————————  step：3  ——————————————#
        # 根据预测值和one-hot的标签计算loss，选择softmax_cross_entropy损失函数；
        
        # Camvid 数据集的weight
        #weights = [0.01,0.007,0.161,0.007,0.02,0.016,0.18, 0.15,0.04,0.29,1.0,0.04]
        #weights = tf.convert_to_tensor(weights)          #将list转成tensor, shape为[50, ]
        #weights = tf.reduce_sum(tf.multiply(one_hot_annotations, weights), -1, name='loss/weights')

        # 采用有weights的loss
        #losses = tf.losses.softmax_cross_entropy(one_hot_annotations, self.predictions, weights=weights, scope='loss/losses')
        
        # 采用普通的loss
        losses = tf.losses.softmax_cross_entropy(one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        
        # PSPnet有特殊的辅助loss
        if self.conf.network_name=="pspnet":
            
            # 采用有weights的loss
            #losses2 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.pred2, weights=weights, scope='loss/losses2')
            
            # 采用普通的loss
            losses2 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.pred2, scope='loss/losses2')
            self.loss_op2 = tf.reduce_mean(losses2, name='loss/loss_op2')
            self.loss_op = self.loss_op + self.loss_op2*0.4
         
        #——————————————  step：4  ——————————————#
        # 选择优化器和学习率，设置训练使用的 train_op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate, 
                beta1=self.conf.beta1, beta2=self.conf.beta2, epsilon=self.conf.epsilon)
        
        # 添加一些需要训练的变量作为train_op依赖项，主要是为了使用BN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        
        #——————————————  step：5  ——————————————#
        # 计算三个评价指标：accuracy、miou、dice，因为没有直接计算dice的函数，
        # 所以保存的是预测值和标签，在验证时再用他们计算dice
        # 1）计算accuracy
        self.decoded_predictions = tf.argmax(self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(self.annotations, self.decoded_predictions, name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        # 2）计算miou
        weights = tf.cast(tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(self.annotations, self.decoded_predictions, self.conf.class_num,
            weights, name='m_iou/m_ious')
        # 3）计算dice----保存需要用的gt和out
        self.out = tf.cast(self.decoded_predictions, tf.float32)
        self.gt = tf.cast(self.annotations, tf.float32)
        
        #——————————————  step：6  ——————————————#
        # 初始化全局变量，这一步需要在session运行训练之前做
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        
        #——————————————  step：7  ——————————————#
        # 用于保存模型和summary
        # 保存BN中不可训练的参数，自己去找那些参数
        trainable_vars = tf.trainable_variables()   #可训练的参数
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'batch_norm/moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'batch_norm/moving_variance' in g.name]
        trainable_vars += bn_moving_vars
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
#———————————————————————————— configure_networks_multi —————————————————————————#       
    """
    函数功能：当使用多个GPU时的网络配置
    输入：无
    输出：无
    """
    def configure_networks_multi(self):
        
        #——————————————  step：1  ——————————————#
        # 设置X\Y的容器；把标签转换成one-hot形式，为了下一步计算loss时使用；
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(tf.int64, self.output_shape, name='annotations')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        expand_annotations = tf.expand_dims(self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(expand_annotations, axis=[self.channel_axis],name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        
        #——————————————  step：2  ——————————————#
        # 利用list记录每个GPU上的指标，然后concat成一个batch后再计算；
        tower_grads = []
        tower_predictions = []
        tower_rate = []
        
        #——————————————  step：3  ——————————————#——设置优化器———#
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate, 
                beta1=self.conf.beta1, beta2=self.conf.beta2, epsilon=self.conf.epsilon)
        
        #——————————————  step：4  ——————————————#——多个GPU的计算———#
        # tf.variable_scope 作用：指定变量的作用域，用于变量共享
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.conf.gpu_num):
                print("this is %d gpu" % i)
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        
                        # 拆分数据给每个GPU；并把标签转换成one-hot形式，为了下一步计算loss时使用；
                        self.x = self.inputs[i * self.conf.batch:(i + 1) * self.conf.batch]
                        self.y = self.annotations[i * self.conf.batch:(i + 1) * self.conf.batch]
                        expand_y = tf.expand_dims(self.y, -1, name='y/expand_dims')
                        one_hot_y = tf.squeeze(expand_y, axis=[self.channel_axis],name='y/squeeze')
                        one_hot_y = tf.one_hot(one_hot_y, depth=self.conf.class_num,
                            axis=self.channel_axis, name='y/one_hot')
                            
                        # 计算预测出来的Y
                        if self.conf.network_name=="ascnet":
                            model = Ascnet(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates = model.inference(self.inputs)
                        if self.conf.network_name=="segnet":
                            model = Segnet(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates = model.inference(self.inputs)
                        if self.conf.network_name=="deeplabv3":
                            model = Deeplabv3(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates = model.inference(self.inputs)
                        if self.conf.network_name=="deeplabv3plus":
                            model = Deeplabv3plus(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates = model.inference(self.inputs)
                        if self.conf.network_name=="unet":
                            model = Unet(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates = model.inference(self.inputs)
                        if self.conf.network_name=="pspnet":
                            model = Pspnet(self.sess, self.conf, self.is_train)
                            self.predictions, self.rates, self.pred2 = model.inference(self.inputs)
                        
                        # 计算loss
                        # Camvid 数据集的weight
                        #weights = [0.01,0.007,0.161,0.007,0.02,0.016,0.18, 0.15,0.04,0.29,1.0,0.04]
                        #weights = tf.convert_to_tensor(weights)          #将list转成tensor, shape为[50, ]
                        #weights = tf.reduce_sum(tf.multiply(one_hot_annotations, weights), -1, name='loss/weights')
                        
                        # 采用有weights的loss
                        #losses = tf.losses.softmax_cross_entropy(one_hot_y, prediction, weights=weights, scope='loss/losses')
                        
                        # 采用普通的loss
                        losses = tf.losses.softmax_cross_entropy(one_hot_y, prediction, scope='loss/losses')
                        loss_each = tf.reduce_mean(losses, name='loss/loss_each')
                        
                        if self.conf.network_name=="pspnet":
                            # 采用有weights的loss
                            #losses2 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.pred2, weights=weights,                                                 scope='loss/losses2')
                            
                            # 采用普通的loss
                            losses2 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.pred2, scope='loss/losses2') 
                            loss_each2 = tf.reduce_mean(losses2, name='loss/loss_op2')
                            loss_each = loss_each + loss_each2*0.4
                            
                        # 共享变量：在第一次声明变量之后，将控制变量重用的参数设置为True，这样可以让不同的GPU更新同一组参数
                        # 注意tf.name_scope函数并不会影响tf.get_variable的命名空间，它只影响tf.variable的
                        tf.get_variable_scope().reuse_variables()
                        
                        # 计算梯度；并保存当前GPU上的指标；
                        grads = optimizer.compute_gradients(loss_each)
                        tower_grads.append(grads)
                        tower_predictions.append(prediction)
                        tower_rate.append(rate)
        
        #——————————————  step：5  ——————————————#
        # 计算平均梯度；并设置训练OP
        grads = self.average_gradients(tower_grads)
        # 添加一些需要训练的变量作为train_op依赖项
        # optimizer.apply_gradients作用：在计算完梯度后，最小化梯度的操作，相当于optimizer.minimize的第二步
        # 添加一些需要训练的变量作为train_op依赖项，主要是为了使用BN
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(grads, name='train_op')
        
        #——————————————  step：6  ——————————————#
        # 计算评价指标
        # 1）计算self.predictions 和 self.rates
        for i in range(self.conf.gpu_num):
            if i==0:
                preds = tower_predictions[i]
                r = tower_rate[i]
            else:
                preds = tf.concat([preds, tower_predictions[i]], self.batch_axis, name='preds/concat'+str(i))
                r = tf.concat([r, tower_rate[i]], self.batch_axis, name='r/concat'+str(i))
        self.predictions = preds
        self.rates = r
        # 2）计算loss
        loss_merge = tf.losses.softmax_cross_entropy(one_hot_annotations, self.predictions, scope='loss/loss_merge')
        self.loss_op = tf.reduce_mean(loss_merge, name='loss/loss_op')
        # 3）计算accuracy
        self.decoded_predictions = tf.argmax(self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(self.annotations, self.decoded_predictions, name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
                    name='accuracy/accuracy_op')
        # 4）计算miou
        weights = tf.cast(tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
                    tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(self.annotations, self.decoded_predictions, self.conf.class_num,
                    weights, name='m_iou/m_ious')
        # 5）计算dice
        self.out = self.decoded_predictions
        self.gt = self.annotations
        
        #——————————————  step：7  ——————————————#——初始化全局变量———#
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        
        for v in tf.global_variables(): 
            print (v.name)
        
        #——————————————  step：8  ——————————————#
        # 用于保存模型和summary
        
        # 保存BN中不可训练的参数，自己去找那些参数
        trainable_vars = tf.trainable_variables()   #可训练的参数
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'batch_norm/moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'batch_norm/moving_variance' in g.name]
        trainable_vars += bn_moving_vars
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        
        # 最原始的保存方式：只保存可训练参数
        #trainable_vars = tf.trainable_variables()
        #self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        #self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
#———————————————————————————— average_gradients —————————————————————————# 
    """
    函数功能：为在所有Tower上共享的变量计算平均梯度
    输入：一个list，其中元素为针对每个GPU的list，内层list元素为（梯度-变量）(gradient, variable)组成的二元元组tuple
    eg. [[(grad0_gpu0,var0_gpu0),(grad1_gpu0,var1_gpu0)],[(grad0_gpu1,var0_gpu1),(grad1_gpu1,var1_gpu1)]]
    输出：一个list，其中元素为（梯度-变量）(gradient, variable)组成的二元元组tuple，其中的梯度是平均值
    """
    def average_gradients(self, tower_grads):
        
        average_grads = []
        
        # grad_and_vars代表不同的参数（含全部GPU），如4个GPU上对应W1的所有梯度值
        # zip 循环在这里的作用：[[(1,2),(3,4)],[(5,6),(7,8)]]转变为了[((1, 2), (5, 6)),((3, 4), (7, 8))]
        # 所以grad_and_vars包含多个元组，每个元组内元素为不同GPU上的某个变量的所有梯度和其变量名
        # [((grad0_gpu0,var0_gpu0),(grad0_gpu1,var0_gpu1)),((grad1_gpu0,var1_gpu0),(grad1_gpu1,var1_gpu1))]
        for grad_and_vars in zip(*tower_grads):
            grads = []
            
            # g就是某一个变量在所有GPU上的梯度的遍历，循环的是不同的GPU
            for g, _ in grad_and_vars:
            
                # 扩展一个维度代表GPU，eg. 如w1=shape(5,10), 扩展后变为shape(1,5,10)
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            # 在第一个维度（也就是代表GPU的维度上）合并，并求平均，这个平均值是某一个变量在所有GPU上梯度的平均值
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
        
            # 注意的是：变量是重复、过剩的，因为每个GPU存储梯度时都存储了变量名
            # 所以我们只把第一个GPU上的变量名抄过来就行了,v是变量名
            v = grad_and_vars[0][1]
            
            # 把平均梯度和变量对应起来
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
#———————————————————————————— train —————————————————————————# 
    """函数功能：模型训练"""   
    def train(self):
       
        # 用于记录summary
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        
        # 有时可以从已训练好的model开始训练
        if self.conf.reload_epoch > 0:
            self.reload(self.conf.reload_epoch)
     
        # 读取数据
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)
        
        # 记录loss
        valid_loss_list = []
        train_loss_list = []
        
        # 记录accuracy
        train_acc_list = []
        valid_acc_list = []
        
        # 记录m_iou
        train_miou_list = []
        valid_miou_list = []
        
        # 初始化局部变量是为了保存训练中的 miou， 因为这是个局部变量
        self.sess.run(tf.local_variables_initializer())
        
        # 开始训练
        for epoch_num in range(self.conf.max_epoch):
            
            # 训练到test_step，在验证集上进行一次验证
            if epoch_num % self.conf.test_step == 1:
                inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, summary = self.sess.run([self.loss_op, self.valid_summary], feed_dict=feed_dict)
                loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)
               
                print(epoch_num, '----valid loss', loss)
                
                # 记录验证集上的loss
                valid_loss_list.append(loss)
                np.save(self.conf.record_dir+"valid_loss.npy",np.array(valid_loss_list))
                # 记录验证集上的acc
                valid_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"valid_acc.npy",np.array(valid_acc_list))
                # 记录验证集上的miou
                valid_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"valid_miou.npy",np.array(valid_miou_list))
                
                ################################### 还是要做训练的呀 #######################################
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                haha, loss, accuracy, m_iou, _ = self.sess.run([self.train_op, self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
                
                print(epoch_num, '----train loss', loss)
                
                # 记录训练集上的loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
                # 记录训练集上的acc
                train_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"train_acc.npy",np.array(train_acc_list))
                # 记录训练集上的miou
                train_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"train_miou.npy",np.array(train_miou_list))
                
            elif epoch_num % self.conf.summary_step == 1:
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, _, summary = self.sess.run([self.loss_op, self.train_op, self.train_summary], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)
                #print(epoch_num)
                
                # 记录训练集上的loss
                #train_loss_list.append(loss)
                #np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
            else:
                
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                loss,_ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
                
                print(epoch_num)
            
            # 保存模型
            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)
#———————————————————————————— test —————————————————————————#   
    # 函数功能：模型验证
    def test(self,model_i):
         
        print('---->testing ', model_i)
        
        # 加载模型
        if model_i > 0:
            self.reload(model_i)
        else:
            print("please set a reasonable test_epoch")
            return
        
        # 读取数据，注意是False，代表不是在训练
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data,False)
        self.sess.run(tf.local_variables_initializer())
       
        # 记录测试参数
        losses = []
        accuracies = []
        m_ious = []
        dices = []
        count = 0
        while True:
            inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
           
            # 终止条件：当取出的batch不够个数了就break
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print(count)
            print('values----->', loss, accuracy, m_iou)          
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            
            # 其实是每一个batch上计算一次指标，最后求均值
            
            out, gt = self.sess.run([self.out, self.gt], feed_dict=feed_dict)
            
            # 只能在二分类时计算dice
            if self.conf.class_num==2:
                tp = np.sum(out*gt)
                fenmu = np.sum(out)+np.sum(gt)+0.000001
                dice = 2*tp/fenmu
            else:
                dice = 1
            
            dices.append(dice)
            print('dice----->', dice)
            count+=1
            if count==self.conf.valid_num:
                break
            
        return np.mean(losses),np.mean(accuracies),m_ious[-1],np.mean(dices)
#———————————————————————————— predict —————————————————————————# 
    # 模型预测
    def predict(self):
         
        print('---->predicting ', self.conf.test_epoch)
        
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        
        # 读取数据
        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())
        predictions = []
        probabilitys = []
        losses = []
        accuracies = []
        m_ious = []
        
        rate_list = []
        count = 0
     
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batchsize)
            
            # 终止条件
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _= self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
            # 记录指标
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            
            # 记录预测值
            predictions.append(self.sess.run(self.decoded_predictions, feed_dict=feed_dict))
            probabilitys.append(self.sess.run(self.predictions, feed_dict=feed_dict))
            
            # 若是使用ASC的话，保存rate field
            if self.conf.use_asc==True:
                rate_list.append(self.sess.run(self.rates, feed_dict=feed_dict))
                
            count+=1
            if count==self.conf.test_num:
                break
        
            
        if self.conf.use_asc==True:
            print('----->saving rate field')
            print(np.shape(rate_list))
            num=0
            for index, prediction in enumerate(rate_list):
            
                # 把一通道的预测值保存为三通道图片，这是自己写的函数
                for i in range(prediction.shape[0]):
                    np.save(self.conf.sample_dir+"rate"+str(num)+".npy",prediction[i])
                    num += 1
                    imsave(prediction[i,:,:,0], self.conf.sample_dir + str(index*prediction.shape[0]+i)+'_rate.png')
                  
        print('----->saving probabilitys')
        print(np.shape(probabilitys))
        np.save(self.conf.sample_dir+"probabilitys"+".npy",np.array(probabilitys))
                     
        print('----->saving predictions')
        print(np.shape(predictions))
        num=0
        for index, prediction in enumerate(predictions):
            
            # 下面的程序用于输出一通道的预测值，测试时需要观察的
            #print(prediction.shape)
            #print(index)
            #np.save("pred",np.array(prediction))
            
            # 把一通道的预测值保存为三通道图片，这是自己写的函数
            for i in range(prediction.shape[0]):
                np.save(self.conf.sample_dir+"pred"+str(num)+".npy",prediction[i])
                num += 1
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i)+'.png')
                
        # 验证和测试的时候，指标都是返回的全体上的均值
        return np.mean(losses),np.mean(accuracies),m_ious[-1]
#———————————————————————————— config_summary —————————————————————————#     
    # 用来配置保存summary
    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.image(name+'/input', self.inputs, max_outputs=100))
        summarys.append(tf.summary.image(name + '/annotation', tf.cast(tf.expand_dims(
                self.annotations, -1), tf.float32), max_outputs=100))
        summarys.append(tf.summary.image(name + '/prediction', tf.cast(tf.expand_dims(
                self.decoded_predictions, -1), tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary
#———————————————————————————— save_summary —————————————————————————# 
    # 保存summary
    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)
#———————————————————————————— save —————————————————————————# 
    # 用来保存模型
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)
#———————————————————————————— reload —————————————————————————# 
    # 用于加载模型
    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
