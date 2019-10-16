import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
import ops

"""
Ascnet类功能：搭建ASCNet网络
函数inference()：用于搭建网络
函数learn_rate_block()：用来学习rate field
函数down_conv_func()等：提取自己在main.py中定义的操作名称

额外注意事项：
1. 为了简化实验，卷积层的通道数都设为相同的；
2. 关于每个网络框架独特的参数设置，在其相应的class内，而不在main.py中；
3. 虽然使用不同卷积的输出不完全一致，为了简便，我们强令rate field一直存在，区别只在于predict时对self.rates调用与否；
"""
class Ascnet(object):
    
    def __init__(self, sess, conf, is_train):
        
        # 设置class内会用到的参数
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
           
    """
    函数功能：用于预测Y，搭建真正的网络结构
    输入：原始图片
    输出：预测图和其他想保存的参数
    """
    def inference(self, inputs):
        
        #####  step：1  #####
        # 学习适合图片的 rate_field
        rate_field = self.learn_rate_block(inputs, 'learnrate') if self.conf.use_asc else inputs
        
        #####  step：2  #####
        # outputs就是每层操作的对象了，为方便使用不改名称
        outputs = inputs
        
        #####  step：3  #####
        # 搭建网络
        for layer_index in range(self.conf.network_depth):
            
            name = 'down%s' % layer_index
            
            # 记录是否是最后一层
            is_final = True if layer_index == (self.conf.network_depth-1) else False
            
            if is_final:
                outputs = self.down_conv_func()(outputs, rate_field, self.conf.class_num, self.conv_size, name, is_train=self.is_train)
                print("down ",layer_index," shape ", outputs.get_shape())
            else:
                outputs = self.down_conv_func()(outputs, rate_field, self.conf.start_channel_num, self.conv_size, name, is_train=self.is_train)
                print("down ",layer_index," shape ", outputs.get_shape())
          
        return outputs,rate_field

    
    """函数功能：用来学习 rate_field 的block"""
    def learn_rate_block(self, inputs, name):
        rate_field = inputs
        rate1 = self.learn_rate_conv_func()(inputs, rate_field, 8, self.conv_size, name+'/conv1', is_train=self.is_train)
        rate2 = self.learn_rate_conv_func()(rate1, rate_field, 4, self.conv_size, name+'/conv2', is_train=self.is_train)
        rate3 = self.learn_rate_conv_func()(rate2, rate_field, 1, self.conv_size, name+'/conv3', is_train=self.is_train)
        
        return rate3
    
    
    # 得取自己定义的卷积和反卷积函数
    def learn_rate_conv_func(self):
        return getattr(ops, self.conf.learn_rate_conv_name)
    
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

    