import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
import ops

"""
Unet类功能：搭建常用的U-Net网络
函数inference()：用于搭建网络
函数construct_down_block()：搭建下采样层
函数construct_bottom_block()：搭建顶层
函数construct_up_block()：搭建上采样层
函数learn_rate_block()：用来学习rate field
函数down_conv_func()等：提取自己在main.py中定义的操作名称

额外注意事项：
1. 为了控制变量做对比实验，卷积层的通道数有两个选择：都相同的；经典U-Net中的倍数增加设置；
2. 关于每个网络框架独特的参数设置，在其相应的class内，而不在main.py中；
3. 虽然使用不同卷积的输入不完全一致，为了简便，我们强令rate field一直存在，区别只在于predict时对self.rates调用与否；
"""
class Unet(object):
    
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
        print("———————————————unet——begin————————————————")
        
        #——————————————  step：2  ——————————————#——下采样部分———#
        # 搭建下采样down网络
        for layer_index in range(self.conf.network_depth-1):
            
            # 记录是否是第一层
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index     # name用来划分定义空间，有利于graph的清晰
           
            # 下采样层
            outputs = self.construct_down_block(outputs, name, down_outputs, rate_field, first=is_first)
            print("down ",layer_index," output_shape: ", outputs.get_shape(),"output_stride: ", self.conf.height//outputs.shape[1].value) 
            print("———————————————(￣︶￣)——————————————————")
             
        #——————————————  step：3  ——————————————#——bottom顶层———#
        # 搭建bottom顶层
        outputs = self.construct_bottom_block(outputs, rate_field, 'bottom')
        print("bottom shape",outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
        print("———————————————(￣︶￣)——————————————————")
        
        #——————————————  step：4  ——————————————#——上采样up层———#
        # 搭建上采样up层
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            
            # 记录是否是最后一层
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            
            # 提取出相应encoder层的输出，以作concat
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_up_block(outputs, down_inputs, name, rate_field, final=is_final)
            print("up ",layer_index," shape ",outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
            print("———————————————(￣︶￣)——————————————————")
            
        print("———————————————unet——end————————————————")
              
        return outputs,rate_field

    """函数功能：搭建下采样层"""
    def construct_down_block(self, inputs, name, down_outputs, rate_field, first=False):
        
        # 计算本层需要输出的filters深度数目
        if self.channel_num_same:
            num_outputs = self.conf.start_channel_num
        else:
            num_outputs = self.conf.start_channel_num if first else 2*inputs.shape[self.channel_axis].value
        
        # encoder每一层包含2个卷积，1个池化
        if first:
            conv1= self.bottom_conv_func()(inputs, rate_field, num_outputs, self.conv_size, name+'/conv1', is_train=self.is_train, bias=False)
            conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, name+'/conv2', is_train=self.is_train, bias=False)
        else:
            conv1= self.bottom_conv_func()(inputs, rate_field, num_outputs, self.conv_size, name+'/conv1', is_train=self.is_train, bias=False)
            conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, name+'/conv2', is_train=self.is_train, bias=False)
        down_outputs.append(conv2)
        pool = ops._max_pool2d(conv2, self.pool_size, name+'/pool')
        return pool

    """函数功能：搭建顶层"""
    def construct_bottom_block(self, inputs, rate_field, name):
        
        if self.channel_num_same:
            num_outputs = self.conf.start_channel_num
            conv1 = self.bottom_conv_func()(inputs, rate_field, num_outputs, self.conv_size, name+'/conv1', is_train=self.is_train, bias=False)
            conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, name+'/conv2', is_train=self.is_train, bias=False)
        else:
            num_outputs = inputs.shape[self.channel_axis].value
            conv1 = self.bottom_conv_func()(inputs, rate_field, 2*num_outputs, self.conv_size, name+'/conv1', is_train=self.is_train, bias=False)
            conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, name+'/conv2', is_train=self.is_train, bias=False)
       
        return conv2

    """函数功能：搭建上采样层"""
    def construct_up_block(self, inputs, down_inputs, name, rate_field, final = False):
        
        if self.channel_num_same:
            num_outputs = self.conf.start_channel_num
            
            # 上采样层包含1个反卷积和2个卷积
            conv1 = self.deconv_func()(inputs, num_outputs, self.conv_size, name+'/conv1')
            conv1 = tf.concat([conv1, down_inputs], self.channel_axis, name=name+'/concat')
         
            if final:
                
                conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, 
                                                name+'/conv2', is_train=self.is_train, bias=False)
                rate_field = ops.conv2d(conv2, rate_field, 1, (3, 3), scope='learnrate2'+'/conv1', rate=1, is_train=self.is_train, bias=False)
                conv3 = self.down_conv_func()(conv2, rate_field, num_outputs, self.conv_size, 
                                   name+'/conv3', is_train=self.is_train, bias=False)
                conv3 = ops.conv2d(conv3, rate_field, self.conf.class_num, (1,1), 
                                   name+'/conv4', is_train=self.is_train, bias=False)
            else:
                conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, 
                                                name+'/conv2', is_train=self.is_train, bias=False)
                conv3 = self.bottom_conv_func()(conv2, rate_field, num_outputs, self.conv_size, 
                                   name+'/conv3', is_train=self.is_train, bias=False)
        else:
            num_outputs = inputs.shape[self.channel_axis].value
            conv1 = self.deconv_func()(inputs, num_outputs, self.conv_size, name+'/conv1')
            conv1 = tf.concat([conv1, down_inputs], self.channel_axis, name=name+'/concat')
            conv2 = self.bottom_conv_func()(conv1, rate_field, num_outputs, self.conv_size, name+'/conv2', is_train=self.is_train, bias=False)
       
            # 计算本层需要输出的filters深度数目
            num_outputs = self.conf.class_num if final else num_outputs/2
            conv3 = self.bottom_conv_func()(conv2, rate_field, num_outputs, self.conv_size, name+'/conv3', is_train=self.is_train, bias=False)
        
        return conv3
    
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