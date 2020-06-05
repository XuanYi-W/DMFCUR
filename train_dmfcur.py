import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import datetime
import DMFCUR_concat


tf.flags.DEFINE_string("word2vec", "data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")


#music
tf.flags.DEFINE_string("valid_data","data/music/dmfcur/data.test", " Data for validation")
tf.flags.DEFINE_string("para_data", "data/music/dmfcur/data.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "data/music/dmfcur/data.train", "Data for training")
tf.flags.DEFINE_string("weight_user", "data/music/dmfcur/W_user.pk", "word2vec file from user vocabulary")
tf.flags.DEFINE_string("weight_item", "data/music/dmfcur/W_item.pk", "Word2vec file from item vocabulary")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs ")



tf.flags.DEFINE_string("rating_matrix","data/music/dmfcur/rating_matrix", " rating matrix ")
tf.flags.DEFINE_string("user_dmf_layer", "512,64", "userlayer filter sizes")
tf.flags.DEFINE_string("item_dmf_layer", "1024,64", "itemlayer filter sizes")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(input_dmf_u, input_dmf_i, input_dmf_y, u_batch, i_batch, uid, iid, reuid, reiid, y_batch,batch_num):
    """
    A single training step
    u_batch, i_batch,
    单词序号列表[array([[1,2,3],[2,5,6],[7,5,6]]),array([[],[],[]]),array([[],[],[]]),...]
    维度 由外向内  用户个数  评论个数  评论长度
    uid, iid,  训练集用户ID array  商品ID array
    reuid, reiid 每个用户交互的商品列表  每个商品交互的用户列表  array
    y_batch 训练集label

    """
    feed_dict = {
        deep.input_dmf_y: input_dmf_y,
        deep.input_dmf_u: input_dmf_u,
        deep.input_dmf_i: input_dmf_i,

        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,

        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return accuracy, mae, u_a, i_a, fm


def dev_step(input_dmf_u, input_dmf_i, input_dmf_y, u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {

        deep.input_dmf_y: input_dmf_y,
        deep.input_dmf_u: input_dmf_u,
        deep.input_dmf_i: input_dmf_i,

        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    return [loss, accuracy, mae]

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS.flag_values_dict()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...Good luck!")
    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_total_num = para['user_total_num']
    item_total_num = para['item_total_num']
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    np.random.seed(2020)
    random_seed = 2020

    print ("user_total_num:",user_total_num)
    print ("item_total_num:",item_total_num)
    print ("user_num:",user_num)
    print ("item_num:",item_num)
    print ("review_num_u:",review_num_u)
    print ("review_len_u:",review_len_u)
    print ("review_num_i:",review_num_i)
    print ("review_len_i:",review_len_i)

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)


        with sess.as_default():
            deep = DMFCUR_concat.DMFCUR_concat(

                user_total_num = user_total_num,
                item_total_num = item_total_num,
                user_dmf_layer = list(map(int, FLAGS.user_dmf_layer.split(","))),
                item_dmf_layer=list(map(int, FLAGS.item_dmf_layer.split(","))),

                review_num_u=review_num_u,
                review_num_i=review_num_i,
                review_len_u=review_len_u,
                review_len_i=review_len_i,
                user_num=user_num,
                item_num=item_num,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                embedding_id=32,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                attention_size=32,
                n_latent=32)
            tf.set_random_seed(random_seed)


            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            ############################

            if FLAGS.rating_matrix:
                init_rating_matrix = np.zeros([user_total_num, item_total_num], dtype=np.float32)
                rating_matrix_ = open(FLAGS.rating_matrix, 'rb')
                init_rating_matrix = pickle.load(rating_matrix_)
                sess.run(deep.rating_matrix.assign(init_rating_matrix))
                print("Ratings_matrix load done!Good luck!\n")

            ##########################
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                W_u_file = open(FLAGS.weight_user, 'rb')
                initW = pickle.load(W_u_file)
                sess.run(deep.W1.assign(initW))

                # load any vectors from the word2vec
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                W_i_file = open(FLAGS.weight_item, 'rb')
                initW = pickle.load(W_i_file)
                sess.run(deep.W2.assign(initW))
                print("Word2vector load done!Good luck!\n")
            #########################

            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0

            # 加载训练数据
            pkl_file = open(FLAGS.train_data, 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.array(train_data)
            pkl_file.close()

            # 加载验证集合
            pkl_file = open(FLAGS.valid_data, 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()


            data_size_train = len(train_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size
            ll = int(len(train_data) / batch_size)

            rmse_ = []
            mae_ = []
            loss_ = []

            for epoch in range(20):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                begin = datetime.datetime.now()

                for batch_num in range(ll):

                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid_dmf, iid_dmf, y_dmf, uid, iid, reuid, reiid, y_batch = zip(*data_train)

                    #***********************dmf
                    uid_dmf = np.array(uid_dmf)
                    iid_dmf = np.array(iid_dmf)
                    y_dmf = np.array(y_dmf)
                    #************************

                    u_batch = []
                    i_batch = []
                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)


                    t_rmse, t_mae, u_a, i_a, fm = train_step(uid_dmf, iid_dmf, y_dmf, u_batch, i_batch, uid, iid, reuid, reiid, y_batch,batch_num)

                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae

                end = datetime.datetime.now()
                print("epoch: " + str(epoch) + " " + "time per epoch: ", str(end-begin) + ":\n")
                print ("train rmse = {}   train mae = {}".format(train_rmse / ll, train_mae / ll))

                print("\nStart evaluation.Good luck !\n")

                train_rmse = 0
                train_mae = 0

                loss_s = 0
                rmse_s = 0
                mae_s = 0


                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    uid_dmf_valid, iid_dmf_valid, y_dmf_valid, userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)

                    #***********************dmf
                    uid_dmf_valid = np.array(uid_dmf_valid)
                    iid_dmf_valid = np.array(iid_dmf_valid)
                    y_dmf_valid = np.array(y_dmf_valid)
                    #************************

                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, rmse, mae = dev_step(uid_dmf_valid, iid_dmf_valid, y_dmf_valid, u_valid, i_valid, userid_valid, itemid_valid, reuid, reiid, y_valid)
                    
                    loss_s = loss_s + len(u_valid) * loss
                    rmse_s = rmse_s + len(u_valid) * np.square(rmse)
                    mae_s = mae_s + len(u_valid) * mae


                loss = loss_s / test_length
                rmse = np.sqrt(rmse_s / test_length)
                mae = mae_s / test_length

                loss_.append(loss)
                rmse_.append(rmse)
                mae_.append(mae)

                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae

                print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss, rmse, mae))

            record = pd.DataFrame(data = [rmse_, mae_, loss_])

            # record.T.to_csv("output/dmfcur/music.csv",header=0)

            print ('best rmse:', best_rmse)
            print ('best mae:', best_mae)
            print("instruments done")
