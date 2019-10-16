import tensorflow as tf
import numpy as np
from deformable_convolution import *
from adaptive_scale_convolution import *

"""
ops.py主要包含网络的各种基础层：
1）卷积操作
2）BN操作
3）activation操作
4）pooling操作
5）upsampling操作
注意：同一种类型的操作必须拥有完全一致的函数输入，这样可保证直接在main中调取的方便；
"""
#######################################################################################################################
######################################   卷积操作   ######################################
####################  包含三种选择：conv2d；adaptive_conv2d；deform_conv2d  ################

"""
函数功能：传统卷积操作；（可选择bias、BN、激活函数）
输入：input feature map；一些参数
输出：output feature map；
"""  
def conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1, 
            is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
    
    ### 卷积层 ###
    # biases_initializer=None 原因：因为一般都会使用BN，BN中设置了center=True，已经有一个偏置β了，就不需要卷积里的bias了；
    # activation_fn=None 原因：使用BN的顺序一般情况是conv+BN+relu，所以这里不需要激活函数；
    # 因为在调用函数时指定scope中一般已经定义了conv做名字，所以使用卷积时直接scope=scope即可；
    
    ### 是否要偏置bias ###
    if bias:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
               data_format=d_format, rate=rate, activation_fn=None, scope=scope)
    else:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
               data_format=d_format, rate=rate, activation_fn=None, biases_initializer=None, scope=scope)
    
    ### BN层 ###
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    
    ### 激活函数 ###
    if activation:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')
    
    return outputs

"""
函数功能：可分离卷积操作；（可选择bias、BN、激活函数）
输入：input feature map；一些参数
输出：output feature map；
"""  
def separable_conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1, 
            is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
    
    ### 卷积层 ###
    # biases_initializer=None 原因：因为一般都会使用BN，BN中设置了center=True，已经有一个偏置β了，就不需要卷积里的bias了；
    # activation_fn=None 原因：使用BN的顺序一般情况是conv+BN+relu，所以这里不需要激活函数；
    # 因为在调用函数时指定scope中一般已经定义了conv做名字，所以使用卷积时直接scope=scope即可；
    
    ### 是否要偏置bias ###
    if bias:
        outputs = tf.layers.separable_conv2d(inputs, num_outputs, kernel_size, strides=stride, padding='same',
               dilation_rate=rate, activation=None, name=scope)
    else:
        outputs = tf.layers.separable_conv2d(inputs, num_outputs, kernel_size, strides=stride, padding='same',
               dilation_rate=rate, activation=None, use_bias= False, name=scope)
    
    ### BN层 ###
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    
    ### 激活函数 ###
    if activation:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')
    
    return outputs

"""
函数功能：自适应尺度卷积操作；（可选择bias、BN、激活函数）
输入：input feature map；一些参数
输出：output feature map；
""" 
def adaptive_conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1, 
                  is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
     
    # 直接输入了rate_field
    rate = rate_field
    
    # 计算倍数
    beishu = rate_field.shape[1].value/inputs.shape[1].value
    if beishu==1:
        rate = rate_field
    else:    
        rate = tf.image.resize_images(rate_field, [inputs.shape[1].value, inputs.shape[2].value])
        rate = tf.div(rate, tf.cast(beishu, tf.float32))
        print("beishu is not 1")
    
    # 生成deformed feature
    input_shape = [inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value]
    asc = ASC(input_shape, kernel_size)
    deformed_feature = asc.adap_conv(inputs, rate, scope)
    
    # 计算实际使用的的步长，必须是kernel_size[0]的倍数
    stride_new = kernel_size[0]*stride
    
    # 完成卷积操作
    if bias:
        outputs = tf.contrib.layers.conv2d(deformed_feature, num_outputs, kernel_size, stride=stride_new,
               padding='VALID', data_format=d_format, rate=1, activation_fn=None, scope=scope)
    else:
        outputs = tf.contrib.layers.conv2d(deformed_feature, num_outputs, kernel_size, stride=stride_new,
               padding='VALID', data_format=d_format, rate=1, activation_fn=None, biases_initializer=None, scope=scope)
    
    ### BN层 ###
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    
    ### 激活函数 ###
    if activation:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')

    return outputs

"""
函数功能：变形卷积操作；（可选择bias、BN、激活函数）
输入：input feature map；一些参数
输出：output feature map；
注意：感受野大小 scope 参数的设置在 deformable_convolution.py 文件里
""" 
def deform_conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1, 
                 is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
    
    # 生成offset-field，原paper中是初始值为0的
    offset = tf.contrib.layers.conv2d(inputs, kernel_size[0]*kernel_size[0]*2, [3,3], scope=scope+'/offset/conv',
          data_format=d_format, activation_fn=None, weights_initializer=tf.zeros_initializer(dtype=tf.float32), biases_initializer=None)
    
    # 进行BN
    offset = tf.contrib.layers.batch_norm(offset, decay=0.9, center=True, activation_fn=tf.nn.tanh,
          epsilon=1e-5, is_training=is_train, scope=scope+'/offset/batch_norm', data_format=d_format)
    
    # 生成deformed feature
    input_shape = [inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value]
    dcn = DCN(input_shape, kernel_size)
    deformed_feature = dcn.deform_conv(inputs, offset, scope)
    
    # 计算实际使用的的步长，必须是kernel_size[0]的倍数
    stride_new = kernel_size[0]*stride
    
    # 完成卷积操作
    if bias:
        outputs = tf.contrib.layers.conv2d(deformed_feature, num_outputs, kernel_size, stride=stride_new,
               padding='VALID', data_format=d_format, rate=1, activation_fn=None, scope=scope)
    else:
        outputs = tf.contrib.layers.conv2d(deformed_feature, num_outputs, kernel_size, stride=stride_new,
               padding='VALID', data_format=d_format, rate=1, activation_fn=None, biases_initializer=None, scope=scope)
    
    ### BN层 ###
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    ### 激活函数 ###
    if activation:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')
    
    return outputs

#######################################################################################################################
###################################   单纯的 BN 操作   ###################################   
"""
函数功能：普通的BN操作
输入：input feature map；一些参数
输出：output feature map；
"""  
def bn(inputs, scope, is_train=True, d_format='NHWC'):
    
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
           epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
   
    return outputs

#######################################################################################################################
###################################   单纯的 activation 操作   ###################################
#########################  包含四种选择：relu；tanh；sigmoid；leaky_relu  ########################

def _relu(inputs, scope):
    
    outputs = tf.nn.relu(inputs, name=scope+'/relu')
   
    return outputs
 
def _tanh(inputs, scope):
    
    outputs = tf.nn.tanh(inputs, name=scope+'/tanh')
   
    return outputs

def _leaky_relu(inputs, scope):
    
    outputs = tf.nn.leaky_relu(inputs, name=scope+'/leaky_relu')
   
    return outputs
 
def _sigmoid(inputs, scope):
    
    outputs = tf.nn.sigmoid(inputs, name=scope+'/sigmoid')
   
    return outputs

#######################################################################################################################
###################################   单纯的 pooling 操作   ###################################
###################  包含三种选择：max_pool2d；avg_pool2d；global_avg_pool2d  ###################

def _max_pool2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    
    outputs = tf.contrib.layers.max_pool2d(inputs, kernel_size, stride=stride, 
           scope=scope+'/max_pool', padding=padding, data_format=data_format)
    
    return outputs

def _avg_pool2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    
    outputs = tf.contrib.layers.avg_pool2d(inputs, kernel_size, stride=stride, 
           scope=scope+'/avg_pool', padding=padding, data_format=data_format)
    
    return outputs

"""
参考网址：https://github.com/mshunshin/SegNetCMR/blob/master/tfmodel/layers.py
"""
def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
    return ret

"""
函数功能：全局平均池化
输入：input feature map；一些参数
输出：output feature map；与input feature map有一样的shape；
"""  
def _global_avg_pool2d(inputs, kernel_size, scope, stride=2, data_format='NHWC'):
    
    # 参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
    outputs = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True, name=scope+'/global_avg_pool')
    
    return outputs

#######################################################################################################################
###################################   单纯的 upsampling 操作   ###################################
#################################   包含两种选择：deconv；bilinear；  #############################
"""
函数功能：反卷积
输入：input feature map；一些参数
输出：output feature map；
""" 
def deconv(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    stride_new = [stride, stride]      # stride代表的是放大的倍数
    outputs = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, scope=scope+'/deconv', stride=stride_new,
           padding='SAME', data_format=d_format, activation_fn=None, biases_initializer=None)
    
    return outputs

"""
函数功能：双线性插值
输入：input feature map；一些参数
输出：output feature map；
""" 
def bilinear(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    size_new = (new_height,new_width)    # 设置新的输出size
    outputs = tf.image.resize_bilinear(inputs, size=size_new, align_corners=True, name=scope+'/bilinear')
    
    return outputs

#######################################################################################################################
###################################   复合的 upsampling 操作   ###################################
"""
函数功能：反卷积+BN+relu
输入：input feature map；一些参数
输出：output feature map；
""" 
def deconv_unit(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    stride_new = [stride, stride]      # stride代表的是放大的倍数
    outputs = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, scope=scope+'/deconv', stride=stride_new,
           padding='SAME', data_format=d_format, activation_fn=None, biases_initializer=None)
    
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    
    return outputs


#######################################################################################################################
#####################################################   垃圾   ######################################################
#######################################################################################################################

