import os
import time
import argparse
import numpy as np
import tensorflow as tf
from actions import Actions

"""
函数功能：设置超参数
"""
def configure():
    
    # 关于训练的参数
    flags = tf.app.flags
    
    #———————————————————————————— 模型设置和训练参数 —————————————————————————# 
    flags.DEFINE_string('network_name', 'segnet', 'Use which framework: segnet, unet, ascnet, deeplabv3, deeplabv3plus, pspnet')
    
    # 这两个参数只有 U-Net和 ASCNet会用
    flags.DEFINE_integer('network_depth', 4, 'network depth')
    flags.DEFINE_integer('start_channel_num', 48, 'start number of outputs')
    
    # 关于训练的参数
    flags.DEFINE_integer('max_epoch', 20000, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 1000, '# of step to test a model')
    flags.DEFINE_integer('save_step', 1000, '# of step to save a model')
    
    # 关于验证的参数
    flags.DEFINE_integer('valid_start_epoch',1,'start step to test a model')
    flags.DEFINE_integer('valid_end_epoch',20001,'end step to test a model')
    flags.DEFINE_integer('valid_stride_of_epoch',1000,'stride to test a model')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    flags.DEFINE_integer('test_epoch', 19001, 'Test or predict epoch')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    
    # 隔一定的epoch就保存summary，是为了使用tensorboard可视化，并不好用，自己画更灵活
    # 所以我把它设置得很大，就不用保存summary啦
    flags.DEFINE_integer('summary_step', 10000000, '# of step to save the summary')
    #———————————————————————————— 训练信息 —————————————————————————# 
    # 设置优化器参数
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1')
    flags.DEFINE_float('beta2', 0.9, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')
 
    # 设置GPU参数
    flags.DEFINE_integer('gpu_num', 1, 'the number of GPU')
    #———————————————————————————— 数据信息 —————————————————————————# 
    flags.DEFINE_string('data_dir', '/home/share/mzhang/data/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'cell_segment2_train.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'cell_segment2_test.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'cell_segment2_test.h5', 'Testing data')
    flags.DEFINE_integer('valid_num',64,'the number of images in the validing set')
    flags.DEFINE_integer('test_num',64,'the number of images in the testing set')
    flags.DEFINE_integer('batch', 2, 'batch size') # 单个GPU上使用的batch
    flags.DEFINE_integer('batchsize', 2, 'total batch size') # 2个GPU上使用的总的batch
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 512, 'height size')
    flags.DEFINE_integer('width', 512, 'width size')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    #———————————————————————————— 存储路径 —————————————————————————#
    flags.DEFINE_string('logdir', '/home/mzhang/TCBB/network1/logdir', 'Log dir')
    flags.DEFINE_string('modeldir', '/home/mzhang/TCBB/network1/modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', '/home/mzhang/TCBB/network1/samples/', 'Sample directory')
    flags.DEFINE_string('record_dir', '/home/mzhang/TCBB/network1/record/', 'Experiment record directory')
    #———————————————————————————— 选择卷积层 —————————————————————————# 
    # 设置使用什么卷积，有四种选择，下采样层、顶层、上采样层可以分别设置
    flags.DEFINE_boolean('use_asc', False, 'use ASC or not')
    flags.DEFINE_string('down_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
    flags.DEFINE_string('bottom_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
    flags.DEFINE_string('up_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
   
    # 设置使用反卷积，不用选择，就用'deconv'
    flags.DEFINE_string('deconv_name', 'deconv', 'Use which deconv op: deconv')
      
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS
#———————————————————————————— train —————————————————————————#
"""
函数功能：训练
"""
def train():
    model = Actions(sess, configure())
    model.train()
#———————————————————————————— valid —————————————————————————#
"""
函数功能：验证
"""
def valid():
    valid_loss = []
    valid_accuracy = []
    valid_m_iou = []
    valid_dice =[]
    conf = configure()
    model = Actions(sess, conf)
    for i in range(conf.valid_start_epoch,conf.valid_end_epoch,conf.valid_stride_of_epoch):
        loss,acc,m_iou,dice=model.test(i)
        valid_loss.append(loss)
        valid_accuracy.append(acc)
        valid_m_iou.append(m_iou)
        valid_dice.append(dice)
        np.save(conf.record_dir+"validate_loss.npy",np.array(valid_loss))
        np.save(conf.record_dir+"validate_accuracy.npy",np.array(valid_accuracy))
        np.save(conf.record_dir+"validate_m_iou.npy",np.array(valid_m_iou))
        np.save(conf.record_dir+"validate_dice.npy",np.array(valid_dice))
        print('valid_loss',valid_loss)
        print('valid_accuracy',valid_accuracy)
        print('valid_m_iou',valid_m_iou)
        print('valid_dice',valid_dice)
#———————————————————————————— predict —————————————————————————#
"""
函数功能：测试
"""
def predict(): 
    predict_loss = []
    predict_accuracy = []
    predict_m_iou = []
    model = Actions(sess, configure())
    loss,acc,m_iou = model.predict()
    predict_loss.append(loss)
    predict_accuracy.append(acc)
    predict_m_iou.append(m_iou)
    print('predict_loss',predict_loss)
    print('predict_accuracy',predict_accuracy)
    print('predict_m_iou',predict_m_iou)
#———————————————————————————— main —————————————————————————#
"""
函数功能：主函数，设置不同的action
"""
def main(argv):
    start = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    # test
    elif args.action == 'test':
        valid()
    # predict
    elif args.action == 'predict':
        predict()
    # train
    else:
        train()
    end = time.clock()
    print("program total running time",(end-start)/60)
#———————————————————————————— GPU设置 —————————————————————————#
if __name__ == '__main__':
    
    # 选择使用哪一块GPU，如果选择多块就是 '2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    # log_device_placement=True 含义：打印设备分配日志
    # allow_soft_placement=True 含义：如果你指定的device不存在或当运行设备不满足要求时，
    # 允许TF自动分配设备，这句话蛮重要，因为有的OP是GPU无法计算的
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    
    # 含义：刚一开始分配少量的GPU容量，然后按需慢慢的增加
    config.gpu_options.allow_growth = True
    
    # 含义：设置每个GPU应该拿出多少容量给进程使用，0.7代表 70%
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    
    # 创建一个符合配置的会话 session
    sess = tf.Session(config=config)
    
    # 处理flag解析，然后执行main函数；如果你的代码中的入口函数叫main()，则你就可以把入口写成tf.app.run()
    # 如果你的代码中的入口函数不叫main()，而是一个其他名字的函数，如test()，则你应该这样写入口tf.app.run(test)
    tf.app.run()
