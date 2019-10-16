import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
import ops

"""
Segnet类功能：搭建常用的segnet网络
函数inference()：用于搭建网络
函数learn_rate_block()：用来学习rate field
函数down_conv_func()等：提取自己在main.py中定义的操作名称

额外注意事项：
1. 为了控制变量做对比实验，卷积层的通道数有两个选择：都相同的；经典U-Net中的倍数增加设置；
2. 关于每个网络框架独特的参数设置，在其相应的class内，而不在main.py中；
3. 虽然使用不同卷积的输入不完全一致，为了简便，我们强令rate field一直存在，区别只在于predict时对self.rates调用与否；
"""
class Segnet(object):
    
    def __init__(self, sess, conf, is_train):
        
        #——————————————  设置参数  ——————————————#
        # 1）传递main中定义的基本参数
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.is_train = is_train
        
        # 2）设置一些需要的参数
        self.data_format = 'NHWC'
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [conf.batch, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]
        self.channel_num_same = True       # 代表通道数数目设置为相同的
#———————————————————————————— 整体的网络架构 —————————————————————————#     
    """
    函数功能：用于预测Y，搭建真正的网络结构
    输入：原始图片
    输出：预测图和其他想保存的参数
    """
    def inference(self, inputs):
        
        #——————————————  step：1  ——————————————#——外挂网络部分———#
        # 学习适合图片的 rate_field
        #rate_field = self.learn_rate_block(inputs, 'learnrate') if self.conf.use_asc else inputs
        rate_field = inputs
        
        
        # outputs就是每层操作的对象了，为方便使用不改名称
        # down_outputs是用来记录下采样层输出，做skip connection的
        outputs = inputs
        down_outputs = []
        print("———————————————segnet——begin————————————————")
        
        #——————————————  step：2  ——————————————#——下采样部分———#
        # 搭建下采样down网络
        
        # 第一层
        name = 'down%s' % 1     # name用来划分定义空间，有利于graph的清晰
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs, arg1 = tf.nn.max_pool_with_argmax(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'maxpool')
        print("down ",1," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第二层
        name = 'down%s' % 2     # name用来划分定义空间，有利于graph的清晰
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs, arg2 = tf.nn.max_pool_with_argmax(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'maxpool')
        print("down ",2," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第三层
        name = 'down%s' % 3     # name用来划分定义空间，有利于graph的清晰
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        outputs, arg3 = tf.nn.max_pool_with_argmax(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'maxpool')
        print("down ",3," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第四层
        name = 'down%s' % 4     # name用来划分定义空间，有利于graph的清晰
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        outputs, arg4 = tf.nn.max_pool_with_argmax(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'maxpool')
        print("down ",4," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第五层
        name = 'down%s' % 5     # name用来划分定义空间，有利于graph的清晰
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        outputs, arg5 = tf.nn.max_pool_with_argmax(outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name+'maxpool')
        print("down ",5," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        #——————————————  step：3  ——————————————#——上采样部分———#
        # 搭建上采样up网络
        
        # 第五层
        name = 'up%s' % 5     # name用来划分定义空间，有利于graph的清晰
        outputs = ops.unpool_with_argmax(outputs, arg5, name='maxunpool5')
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        print("up ",5," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第四层
        name = 'up%s' % 4     # name用来划分定义空间，有利于graph的清晰
        outputs = ops.unpool_with_argmax(outputs, arg4, name='maxunpool4')
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 512, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        print("up ",4," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第三层
        name = 'up%s' % 3     # name用来划分定义空间，有利于graph的清晰
        outputs = ops.unpool_with_argmax(outputs, arg3, name='maxunpool3')
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 256, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        print("up ",3," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第二层
        name = 'up%s' % 2     # name用来划分定义空间，有利于graph的清晰
        outputs = ops.unpool_with_argmax(outputs, arg2, name='maxunpool2')
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        print("up ",2," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        # 第一层
        name = 'up%s' % 1     # name用来划分定义空间，有利于graph的清晰
        outputs = ops.unpool_with_argmax(outputs, arg1, name='maxunpool1')
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv1', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv2', is_train=self.is_train, bias=False)
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
        print("up ",1," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
        print("———————————————(￣︶￣)——————————————————")
        
        name = 'final'
        outputs = ops.conv2d(outputs, rate_field, self.conf.class_num, (3, 3), scope=name+'/conv3', is_train=self.is_train, bias=False)
           
        print("———————————————segnet——end————————————————")
              
        return outputs,rate_field
#———————————————————————————— 外挂网络 —————————————————————————# 
    """函数功能：用来学习 rate_field 的block"""
    def learn_rate_block(self, inputs, name):
        rate_field = inputs
        outputs = ops.conv2d(inputs, rate_field, 8, (3, 3), scope=name+'/conv1', rate=1, is_train=self.is_train, bias=False)
        outputs = ops.conv2d(outputs, rate_field, 4, (3, 3), scope=name+'/conv2', rate=2, is_train=self.is_train, bias=False)
        outputs = ops.conv2d(outputs, rate_field, 1, (3, 3), scope=name+'/conv3', rate=1, is_train=self.is_train, bias=False)
       
        return outputs

    # 得取自己定义的卷积和反卷积函数
    def down_conv_func(self):
        return getattr(ops, self.conf.down_conv_name)
    
    def bottom_conv_func(self):
        return getattr(ops, self.conf.bottom_conv_name)
    
    def up_conv_func(self):
        return getattr(ops, self.conf.up_conv_name)
    
    def add_conv_func(self):
        return getattr(ops, self.conf.add_conv_name)
    
    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)