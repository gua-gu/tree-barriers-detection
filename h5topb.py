import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import load_model


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    """
    .h5模型文件转换成pb模型文件
    :param h5_model: .h5模型
    :param output_dir: pb模型文件保存路径
    :param model_name: pb模型文件名称
    :param out_prefix: 根据训练，需要修改
    :param log_tensorboard: 是否生成日志文件，默认为True
    :return: pb模型文件
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    # 输出日志文件
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, model_name), output_dir)

load_h5_model = load_model('E:\python\shuzhang', custom_objects=keras.utils.generic_utils.get_custom_objects())
h5_to_pb(load_h5_model, output_dir='E:\python\shuzhang', model_name='vgg16+de_shuzhang.pb')
