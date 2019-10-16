import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
import ops

"""
Deeplabv3类功能：搭建Deeplabv3网络
---------------
| 各函数功能：|
---------------
函数inference()：用于搭建网络
-------------------------------- encoder --------------------------------------
函数resnet()：用来搭建resnet，可以选择50、101、152层，这是deeplabV3的encoder部分；
函数start_block()：构建resnet中的conv1；
函数bottleneck()：属于resnet中的一个组件；
-------------------------------- ASPP --------------------------------------
函数ASPP()：构建ASPP模块；
-------------------------------- others --------------------------------------
函数learn_rate_block()：用来学习rate field
函数down_conv_func()等：提取自己在main.py中定义的操作名称
---------------
|  注意事项： |
---------------
1. 为了还原论文，卷积层的通道数按照论文设置；
2. 关于每个网络框架独特的参数设置，在其相应的class内的init函数下，而不在main.py中；
3. 虽然使用不同卷积的输入不完全一致，为了简便，我们强令rate field一直存在，区别只在于predict时对self.rates调用与否；
4. ops文件里的所有卷积都拥有同样的输入；
5. 每个网络的参数设置只能放在init函数下，方便一眼看到；
6. 网络中的1*1卷积直接调用，3*3卷积必须使用down_conv_func()函数间接调用，以保证可以选择想用的卷积类型；
7. 网络中只使用down_conv_func()一种，如果想同时使用两种卷积，请你再新建一份py文件，别改我老本！！！
"""
class Deeplabv3(object):
#———————————————————————————— 请在这里设置参数 @_@ —————————————————————————#
    def __init__(self, sess, conf, is_train):
        
        # 传递main中定义的基本参数
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.is_train = is_train
        
        # 设置基础参数
        self.data_format = 'NHWC'
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [conf.batch, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]
        #——————————————  常用参数，只看这里就行  ——————————————#
        # 设置multi-grid的rate
        self.multi_grid = [1,2,1]
        
        # 选择网络深度：选项为 50,101,152
        self.network_depth = 50
        
        # 设置block深度，可以设成4，也可以设成7
        self.block_depth = 7
        
        # 设置下采样程度：选项为 8,16
        self.output_stride = 16
        #—————————————————————————————————————————#
        # 设置block1-7的输出通道数
        self.num_outputs = [256, 512, 1024, 2048, 2048, 2048, 2048]
        
        # 设置block1-7的下采样率
        if self.output_stride==16:
            self.strides = [2, 2, 1, 1, 1, 1, 1]
            self.block_rate = [1, 1, 1, 2, 4, 8, 16]
        elif self.output_stride==8:
            self.strides = [2, 1, 1, 1, 1, 1, 1]
            self.block_rate = [1, 1, 2, 4, 8, 16, 32]
        else:
            print("The output stride is wrong!")
        
        # 设置每个block内使用多少个bottleneck，给出了7个block的设置，当然也可以只用4个；
        if self.network_depth==50:
            self.bottleneck_num = [3, 4, 6, 3, 3, 3, 3]     # 如果使用resnet-50
        elif self.network_depth==101:
            self.bottleneck_num = [3, 4, 23, 3, 3, 3, 3]    # 如果使用resnet-101
        elif self.network_depth==152:
            self.bottleneck_num = [3, 8, 36, 3, 3, 3, 3]    # 如果使用resnet-152
        else:
            print("The network depth is wrong!")
#———————————————————————————— 整体的网络架构 —————————————————————————#  
    """
    函数功能：用于预测Y，搭建真正的网络结构
    输入：原始图片
    输出：预测图和其他想保存的参数
    """
    def inference(self, inputs):
        
        print("———————————————deeplabV3——begin————————————————")
        
        #——————————————  step：1  ——————————————#——外挂网络部分———#
        # 学习适合图片的 rate_field，保留ASC功能，可以选择不用
        rate_field = self.learn_rate_block(inputs, 'learnrate') if self.conf.use_asc else inputs
        
        #——————————————  step：2  ——————————————#——resnet部分———#
        # outputs就是每层操作的对象了，为方便使用不改名称
        outputs = inputs
        name = 'resnet'
        outputs = self.resnet(outputs, rate_field, name) 
                
        #——————————————  step：3  ——————————————#—— ASPP ———#
        name = 'aspp'
        outputs = self.ASPP(outputs, rate_field, name)
        
        #——————————————  step：4  ——————————————#—— final ———#
        name = 'final'
        outputs = ops.conv2d(outputs, rate_field, self.conf.class_num, (1, 1), scope=name+'/conv_final', is_train=self.is_train, bias=False)
        outputs = tf.image.resize_bilinear(outputs, size=(self.output_shape[1],self.output_shape[2]), align_corners=True, name=name+'/bilinear')
        print(name," output_shape: ", outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
        print("———————————————deeplabV3——end————————————————")
           
        return outputs,rate_field
#———————————————————————————— resnet-50--101--152 —————————————————————————#     
    """
    函数功能：把resnet打包成一个组件
    输入：input feature map
    输出：output feature map
    注意：为了减少参数，我们使用3个3*3卷积代替一个7*7卷积，原文代码中也有这样做的选择；
    """
    def resnet(self, inputs, rate_field, name):
        
        mainname = name
        #——————————————  step：1  ——————————————#——构建conv1———#
        outputs = inputs
        name = mainname+'_start_block'
        outputs = self.start_block(outputs, rate_field, name)
        print(name," output_shape: ", outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
        print("———————————————(￣︶￣)——————————————————")
        
        #——————————————  step：2  ——————————————#——7个block部分———#
        # 需要满足的条件：1）block内最后一个bottle输入需要的stride；2）block4-7 采用multi-grid模式
        for block_index in range(self.block_depth):
            for bottle_index in range(self.bottleneck_num[block_index]):
                name = mainname+'---block%s' % (block_index+1) + '---bottle%s' % (bottle_index+1)
                
                # block内最后一个bottle输入需要的stride
                is_last = True if bottle_index==(self.bottleneck_num[block_index]-1) else False
                stride = self.strides[block_index] if is_last else 1
                
                # block4-7 采用multi-grid模式
                unit_rate = self.multi_grid[bottle_index]if block_index>=3 else 1
                
                # 创建一个 bottleneck
                outputs = self.bottleneck(outputs, rate_field, self.num_outputs[block_index]//4, self.num_outputs[block_index], 
                       name, stride=stride, unit_rate=unit_rate, block_rate=self.block_rate[block_index])
                print(name," output_shape: ", outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
            print("———————————————(￣︶￣)——————————————————")
        
        return outputs
#———————————————————————————— start_block —————————————————————————#     
    """
    函数功能：网络最开始的卷积和池化
    输入：input feature map
    输出：output feature map
    注意：为了减少参数，我们使用3个3*3卷积代替一个7*7卷积，原文代码中也有这样做的选择；
    """
    def start_block(self, inputs, rate_field, name):
        
        # 3个3*3卷积
        # norm=False 意味着 卷积层由 weight+bias+relu组成，没有使用BN
        outputs = self.down_conv_func()(inputs, rate_field, 64, (3, 3), name+'/conv1', stride=2, is_train=self.is_train, norm=False)
        outputs = self.down_conv_func()(outputs, rate_field, 64, (3, 3), name+'/conv2', is_train=self.is_train, norm=False)
        outputs = self.down_conv_func()(outputs, rate_field, 128, (3, 3), name+'/conv3', is_train=self.is_train, norm=False)
        
        # 1个最大池化
        # 注意这个max-pooling的kernel size是（3,3），而不是一般用的（2,2）
        outputs = ops._max_pool2d(outputs, (3, 3), name+'/max_pool')
        
        return outputs
#———————————————————————————— bottleneck —————————————————————————#     
    """
    函数功能：resnet中的bottleneck
    输入：input feature map
    输出：output feature map
    注意：
    1）num_front是指前两个卷积的通道数，num_outputs是指第三个卷积的通道数，eg. 64,64,256
    2）关于stride=2下采样部分，和multi-grid设置都是在第二个3*3卷积上进行
    3）unit_rate是指multi-grid的设置，block_rate是指这个block本身需要的rate，最终的rate=两者乘积
    4）为了加速，残差部分的3个卷积都使用了BN
    """
    def bottleneck(self, inputs, rate_field, num_front, num_outputs, name, stride=1, unit_rate=1, block_rate=1):
        
        #——————————————  step：1  ——————————————#——shortcut部分———#
        # 检查是否满足tf.add的条件，因为输入和输出的tensor必须shape完全一致才能相加
        # 如果输入输出通道数一致，则只需要进行H*W尺寸的统一；否则，使用1*1卷积进行统一
        num_inputs = inputs.shape[self.channel_axis].value    # 输入的tensor通道数
        if num_inputs == num_outputs:
            # 下采样采用的是1*1池化，因为原始文章的代码中subsample函数是使用1*1池化实现的
            shortcut = inputs if stride==1 else ops._max_pool2d(inputs, (1, 1), name+'/max_pool', stride=stride) 
        else:
            shortcut = ops.conv2d(inputs, rate_field, num_outputs, (1, 1), scope=name+'/shortcut', stride=stride, 
                           is_train=self.is_train, norm=False, activation=False)    # 1*1卷积没有使用激活函数和BN
            
        #——————————————  step：2  ——————————————#——residual残差部分——#
        # resnet原文code实现中使用了BN，我们在这里都使用BN、activation，为了加速收敛
        outputs = ops.conv2d(inputs, rate_field, num_front, (1, 1), scope=name+'/conv1', is_train=self.is_train, bias=False)
        # 注意：当需要下采样时，原文采用的是conv2d_same，其实就是普通卷积
        outputs = self.down_conv_func()(outputs, rate_field, num_front, (3, 3), scope=name+'/conv2', stride=stride,
                             rate=unit_rate*block_rate, is_train=self.is_train, bias=False)
        outputs = ops.conv2d(outputs, rate_field, num_outputs, (1, 1), scope=name+'/conv3', 
                      is_train=self.is_train, bias=False, activation=False)
        
        #——————————————  step：3  ——————————————#——加和部分——#
        outputs = ops._relu(tf.add(outputs, shortcut), scope=name)   # scope=name即可是因为函数内部有命名relu
        
        return outputs
#—————————————————————————————— ASPP ———————————————————————————# 
    """
    函数功能：ASPP
    输入：input feature map
    输出：output feature map
    注意：
    1）ASPP内的每一个卷积都使用了BN
    """
    def ASPP(self, inputs, rate_field, name, num_outputs=256):
        
        #——————————————  step：1  ——————————————#——global-avg-pooling部分———#
        # 包括三个部分：池化+卷积+插值
        height = inputs.shape[1].value
        width = inputs.shape[2].value
        global_avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True, name=name+'/global_avg_pool')  
        global_avg_pool = ops.conv2d(global_avg_pool, rate_field, num_outputs, (1, 1), scope=name+'/conv1', is_train=self.is_train, bias=False)
        global_avg_pool = tf.image.resize_bilinear(global_avg_pool, size=(height,width), align_corners=True, name=name+'/bilinear')
        
        #——————————————  step：2  ——————————————#——不同rate的卷积部分———#
        aspp1 = ops.conv2d(inputs, rate_field, num_outputs, (1, 1), scope=name+'/aspp1', is_train=self.is_train, bias=False)
        aspp2 = self.down_conv_func()(inputs, rate_field, num_outputs, (3, 3), scope=name+'/aspp2', rate=6, is_train=self.is_train, bias=False)
        aspp3 = self.down_conv_func()(inputs, rate_field, num_outputs, (3, 3), scope=name+'/aspp3', rate=12, is_train=self.is_train, bias=False)
        aspp4 = self.down_conv_func()(inputs, rate_field, num_outputs, (3, 3), scope=name+'/aspp4', rate=18, is_train=self.is_train, bias=False)
        
        #——————————————  step：3  ——————————————#——concat整合部分———#
        outputs = tf.concat((global_avg_pool, aspp1, aspp2, aspp3, aspp4), axis=3, name=name+'/concat')
        outputs = ops.conv2d(outputs, rate_field, num_outputs, (1, 1), scope=name+'/conv2', is_train=self.is_train, bias=False)
        
        print(name," output_shape: ", outputs.get_shape(), "output_stride: ", self.conf.height//outputs.shape[1].value)
        print("———————————————(￣︶￣)——————————————————")
        
        return outputs
#———————————————————————————— 外挂网络 —————————————————————————# 
    """函数功能：用来学习 rate_field 的block"""
    def learn_rate_block(self, inputs, name):
        rate_field = inputs
        outputs = ops.conv2d(inputs, rate_field, 8, (3, 3), scope=name+'/conv1', rate=1, is_train=self.is_train, bias=False)
        outputs = ops.conv2d(outputs, rate_field, 4, (3, 3), scope=name+'/conv2', rate=2, is_train=self.is_train, bias=False)
        outputs = ops.conv2d(outputs, rate_field, 1, (3, 3), scope=name+'/conv3', rate=1, is_train=self.is_train, bias=False)
       
        return outputs
#———————————————————————————— 得取自己定义的基础层 —————————————————————————#   
    def down_conv_func(self):
        return getattr(ops, self.conf.down_conv_name)
    
    def bottom_conv_func(self):
        return getattr(ops, self.conf.bottom_conv_name)
    
    def up_conv_func(self):
        return getattr(ops, self.conf.up_conv_name)
    
    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)
    