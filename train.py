import tensorflow as tf
import os
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler
from simanneal import Annealer
import random
import sys
from termcolor import  colored
import sqlite3
import datetime
import math
import time
now=datetime.datetime.now()

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_iter', 10000, 'Total training iterations')
flags.DEFINE_string('model_dir', 'C:\\Users\\Mohammad\\PycharmProjects\\final_1D\\test', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', 'C:\\Users\\Mohammad\\Downloads\\data_scene_flow\\training', 'training dataset dir')
flags.DEFINE_string('util_root', 'C:\\Users\\Mohammad\\Downloads\\data_scene_flow', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')

FLAGS = flags.FLAGS

np.random.seed(123)

dhandler = Data_handler(data_version=FLAGS.data_version,
                        data_root=FLAGS.data_root,
                        util_root=FLAGS.util_root,
                        num_tr_img=FLAGS.num_tr_img,
                        num_val_img=FLAGS.num_val_img,
                        num_val_loc=FLAGS.num_val_loc,
                        batch_size=FLAGS.batch_size,
                        patch_size=FLAGS.patch_size,
                        disp_range=FLAGS.disp_range)

if FLAGS.data_version == 'kitti2012':
    num_channels = 1
elif FLAGS.data_version == 'kitti2015':
    num_channels = 3
else:
    sys.exit('data_version should be either kitti2012 or kitti2015')


class SimulatedAnnealer(Annealer):
    def __init__(self, state):
        self.num = 0
        self.best=math.inf
        self.path=FLAGS.model_dir+'\\bests.db'
        #self.path1='C:\\Users\\Mohammad\\Desktop\\version 3\\model1\\all.db'
        conn=sqlite3.connect(self.path)
        c=conn.cursor()
        c.execute('''CREATE TABLE bestss
                     (num int, arc text, acc real, t_flops real, energy real,time real)''')
        conn.commit()

        conn.close()
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                             (num int, arc text, acc real, t_flops real, energy real,time real)''')
        conn.commit()

        conn.close()
        super(SimulatedAnnealer, self).__init__(state)

    def move(self):

        print('////////////////////////////////////////////')
        #print('first len==',len(self.state))
        #print(1)
        valids=[]
        #print(2)
        others=[]
        #print(3)
        layers=[['conv2d',32,'same',3],['conv2d',32,'same',5],['conv2d',32,'same',7],['conv2d',32,'same',11],
                ['conv2d',64,'same',3],['conv2d',64,'same',5],['conv2d',64,'same',7],['conv2d',64,'same',11],['batch',0,'none',0]]
        #print(4)
        for i in self.state:
            #print(5)
            if i[0]=='conv2d':
                #print(6)
                if i[2]=='valid':
                    #print(7)
                    #print('vvvvvvvvvvvvvvvvvvv',i[3])
                    if i[3]>3:
                        #print(8)


                        valids.append(self.state.index(i))
                        #print(9)
                    else:
                        #print('dw')
                        pass
                else:
                    #print(10)
                    others.append(self.state.index(i))
                    #print(11)
            else:
                #print(12)
                others.append(self.state.index(i))
                #print(13)
        if len(valids)!=0:
            #print(14)

            if random.random()<0.2:
                #print(15)
                #print('valid change')

                #print('valids=',valids)
                a=random.choice(valids)
                '''print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                for i in valids:
                    print(self.state[i])
                print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')'''
                #print(16)
                #print('state index',a)
                #print(self.state[a])
                b=random.randrange(3,self.state[a][3],2)
                #print(17)
                #print('kernel',b)

                #print('hrt',b)
                temp=self.state[a][3]
                #print(18)
                #print('temp',temp)
                self.state[a][3]=b
                #act=['relu','elu','swish','leaky_relu','none']
                #ac=random.choice(act)
                #print(19)
                #print('starteter',self.state[a])
                self.state.append(['conv2d',32,'valid',(temp-b)+1])
                #print(20)
                #print(self.state)

            elif random.random()>=0.2 and random.random()<0.7:
                #print(21)
                print('others change')
                '''print('ergthyjukyjtrewq ertyujhtrew')
                for i in others:
                    print(self.state[i])
                print('fwrgthrgegregggergrgrgergegge')'''
                #print(22)
                a=random.choice(others)
                #print(23)
                be=random.choice(layers)
                #print(24)
                #print(a)
                #print(25)
                #print('be',be)
                #print(26)
                #self.state[a]=be
                self.state.remove(self.state[a])
                self.state.append(be)
                #print(27)
            else:
                #print(28)
                print('sweap')
                #print(29)

                a=random.randint(0,len(self.state)-1)
                #print(30)
                b=random.randint(0,len(self.state)-1)
                print(a,b)
                #print(31)
                temp1=self.state[a]
                self.state[a]=self.state[b]
                self.state[b]=temp1
                #self.state[a],self.state[b]=self.state[b],self.state[a]
                #print(32)
        else:
            #print(33)
            print('else')
            #print(34)
            if random.random()<0.5:
                #print(35)
                print('others change')
                #print(36)
                a = random.choice(others)
                #print(37)
                #print(a)
                #print(38)
                be = random.choice(layers)
                #print(39)
                #print('be', be)
                #print(40)
                #self.state[a] = be
                self.state.remove(self.state[a])
                self.state.append(be)
                #print(41)
            else:
                #print(42)
                print('sweap')
                #print(43)
                a = random.randint(0, len(self.state) - 1)
                #print(44)
                b = random.randint(0, len(self.state) - 1)
                #print(45)
                temp1 = self.state[a]
                self.state[a] = self.state[b]
                self.state[b] = temp1
                #self.state[a], self.state[b] = self.state[b], self.state[a]
                #print(46)
        print(len(self.state),self.state)
        kernel_sum=0
        num_node=0
        for i in self.state:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    kernel_sum+=i[3]
                    num_node+=1
        ex=kernel_sum-num_node
        if ex!=36:
            #print('2',self.state)
            self.state.append(['conv2d',32,'valid',(36-ex)+1])
            #raise ValueError('this is not appropriante')
        #print(47)
        print('self state type',type(self.state))

        return self.energy()

    def energy(self):
        #print(self.state)
        kernel_sum=0
        num_node=0
        for i in self.state:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    kernel_sum+=i[3]
                    num_node+=1
        ex=kernel_sum-num_node
        if ex!=36:
            print('2',self.state)
            raise ValueError('this is not appropriante')



        print('self num=',self.num)
        t_flops,time=train(self.state, self.num)
        print('self num=',self.num)
        acc = evaluate(self.state,self.num)
        '''acc=acc/100
        flops=(18700000-t_flops)/18700000
        e=0.44*(1-acc)+0.56*(1-flops)
        if acc>0.5:
            e+=0.1'''
        if acc==0.0:
            e=math.inf
        else:
            e=t_flops/acc

        statea=str(self.state)
        print(colored('rggtrggegergggwgegwgw','yellow'),e)
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?)''', [self.num, statea, acc, t_flops, e,time])
        conn.commit()
        conn.close()

        if e<self.best:
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?,?)''',[self.num,statea,acc,t_flops,e,time])
            conn.commit()
            conn.close()
            self.best=e
            print(colored('4ogmmregreomgerogmreomerormfrofmfoemwfoewmfewofm','red'))
        self.num = self.num + 1
        print(str(now))

        return e
        #return random.random()


def train(state, number):
    path = FLAGS.model_dir + '/' + str(number)
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    g = tf.Graph()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        with g.as_default():

            limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

            snet = nf.create(limage, rimage, targets, state, FLAGS.net_type)

            loss = snet['loss']
            train_step = snet['train_step']
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            acc_loss = tf.placeholder(tf.float32, shape=())
            loss_summary = tf.summary.scalar('loss', acc_loss)
            train_writer = tf.summary.FileWriter(path + '/training', g)

            saver = tf.train.Saver(max_to_keep=1)
            losses = []
            summary_index = 1
            lrate = 1e-2
            total_time=0

            for it in range(1, FLAGS.num_iter):
                lpatch, rpatch, patch_targets = dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                t1=int(round(time.time()*1000))
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                t2 = int(round(time.time() * 1000))
                total_time+=t2-t1
                losses.append(mini_loss)

                if it % 100 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss))
                    saver.save(session, os.path.join(path, 'model.ckpt'), global_step=snet['global_step'])
                    train_summary = session.run(loss_summary,
                                                feed_dict={acc_loss: np.mean(losses)})
                    train_writer.add_summary(train_summary, summary_index)
                    summary_index += 1
                    train_writer.flush()
                    losses = []

                if it == 24000:
                    lrate = lrate / 5.
                elif it > 24000 and (it - 24000) % 8000 == 0:
                    lrate = lrate / 5.
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            t_flops=flops.total_float_ops
            print('wetwyy', t_flops)
    return t_flops,total_time/1000.0


def evaluate(state,number):
    lpatch, rpatch, patch_targets = dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path = FLAGS.model_dir + '/' + str(number)

    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, state,FLAGS.net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], FLAGS.eval_size):
            eval_dict = {limage: lpatch[i: i + FLAGS.eval_size],
                         rimage: rpatch[i: i + FLAGS.eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i + FLAGS.eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

        print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
    return ((acc_count / lpatch.shape[0]) * 100)


'''if FLAGS.phase == 'train':
	train()
elif FLAGS.phase == 'evaluate': 
	evaluate()
else:
	sys.exit('FLAGS.phase = train or evaluate')'''
if __name__ == '__main__':
    init = [['conv2d', 32, 'same',5],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'same',5],
            ['none',0,'none',0],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'same',5],
            ['conv2d', 64, 'valid',37]]
    tsp = SimulatedAnnealer(init)
    #print('///////////////////////////////////////////////////////////////')
    tsp.set_schedule(tsp.auto(0.01,10))
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()
    print()
    print("%i mile rout:" % e)
    print("state {}".format(state))