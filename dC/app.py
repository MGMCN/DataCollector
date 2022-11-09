import datetime  # 导入时间和随机数模块
import random
import hashlib
import threading

from flask import Flask
from flask import request
from flask import jsonify  # 数据转为json，并以字典的形式传回前端
from flask import json
from flask import Response
from flask import render_template
from werkzeug.routing import BaseConverter
from flask_sqlalchemy import SQLAlchemy

from dC.DDQN import DeepQNetwork


class RegexConverter(BaseConverter):
    def __init__(self, url_map, *args):
        super(RegexConverter, self).__init__(url_map)
        self.url = url_map
        self.regex = args[0]  # flask的正则匹配得这么干...


app = Flask(__name__)
app.url_map.converters['re'] = RegexConverter
app.config['SECRET_KEY'] = 'xxxxxx'  # 密码
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:xxxxxx@163.221.29.17:xxxx/test'  # lab Cloud

# db
database = None

# test
nodes = []

# cache 放一些还没组织完全的数据
cache = {}

# 给RL用的 从这个里面拿特定格式的数据需要单独写个def
batch_memory = {}


# nodes = [
#     {'name': 'Node1', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node2', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node3', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node4', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node5', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node6', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node7', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node8', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node9', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node10', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
#     {'name': 'Node11', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
# ]


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html', nodes=nodes)


@app.route('/<re(r"Node[0-9]*"):path>')
def details(path):
    # print(path)
    datasets = None
    nodeId = None
    for node in nodes:
        if node['name'] == path:
            datasets = node['datasets']
            nodeId = node['id']
            break
    return render_template('details.html', nodeName=path, nodeId=nodeId, datasets=datasets)


# 生成实时数据
@app.route('/setData/', methods=['POST'])
def setData():
    jsondata = request.json
    # print('recv :', json['node'], json['key'])

    now = datetime.datetime.now().strftime('%H:%M:%S')
    data = {'time': now, 'data': random.randint(1, 10)}
    return jsonify(data)  # 将数据以字典的形式传回


# 接收节点上传的数据
@app.route('/postData/', methods=['POST'])
def postData():
    global modelActivate, DuelingNetwork, lock
    jsondata = request.json

    node = jsondata['node']
    nodeId = jsondata['id']
    datasets = jsondata['datasets']
    # print('recv :', node)  # datasets : {}
    # for data_type in datasets:
    #     print(data_type, datasets[data_type])

    lock.acquire()
    if not modelActivate:
        print('创建模型')
        DuelingNetwork = DeepQNetwork(input=input_data_length, actions_cnt=3, memory_size=3, batch_size=2,
                                      e_greedy_increment=0.1)
        # do create model function
        modelActivate = True
    lock.release()

    found = False
    for n in nodes:
        if node == n['name']:
            found = True
            break
    if not found:
        n = {'name': node, 'id': nodeId, 'datasets': datasets}
        nodes.append(n)  # 这里是展示的数据 还没写好...

    if node not in batch_memory:
        batch_memory[node] = {}  # batch_memory{ node{} }
        batch_memory[node]['id'] = nodeId  # batch_memory{ node{id:} }
        batch_memory[node]['datasets'] = {}  # batch_memory{ node{id:,datasets:{}} }
        for data_type in datasets:
            batch_memory[node]['datasets'][data_type] = []  # batch_memory{ node{id:,datasets:{type1:[],type2:[]}} }
            batch_memory[node]['datasets'][data_type].append(datasets[data_type])  # []当stack用

    # store data
    # 存数据我们就存个15s的就行了存多了的就移除
    # 访问这个得加锁
    for data_type in datasets:
        if len(batch_memory[node]['datasets'][data_type]) > 15:
            batch_memory[node]['datasets'][data_type].pop(0)
        batch_memory[node]['datasets'][data_type].append(datasets[data_type])  # []当stack用

    return "post succeed!"


DuelingNetwork = None

modelActivate = False

input_data_length = 81  # 9*9

time_duration = 9

lock = threading.Lock()

model_lock = threading.Lock()

selectCnt = 0


# 节点上传想要参与选举的节点列表ID
@app.route('/duelingSelect/', methods=['POST'])
def duelingSelect():
    global modelActivate, DuelingNetwork
    jsondata = request.json

    node = jsondata['node']
    nodeId = jsondata['id']
    neighbors = jsondata['neighbors']
    input_state = []
    # print(nodeId, 'want select ->', neighbors)

    # 模型已经创建好了，等着input数据了，但我们得先整理数据才能input进去
    index = len(nodes) - 1
    # 访问batch_memory的时候可能需要加个锁
    for node_index in range(index):
        for data_type in batch_memory['Node' + str(node_index + 1)]['datasets']:
            datas = batch_memory['Node' + str(node_index + 1)]['datasets'][data_type]
            # print(data_type)
            for data in datas[-time_duration:]:  # 取尾巴9个数据
                input_state.append(data)
            # print('看看这里append的size ->', len(input_state))

    # 把input_state的数据给reshape下就可以输入进去了然后拿到各个节点的评分输出(这里逻辑还没写)
    model_lock.acquire()
    while True:
        output = DuelingNetwork.choose_action(input_state, neighbors, batch_memory)
        if output == -1:
            break
        if 'Node' + str(output) != node:
            break
    model_lock.release()
    # print(node, '选择了 ->', 'Node' + str(output + 1))
    if output != -1:
        idSelected = batch_memory['Node' + str(output)]['id']
    else:
        idSelected = neighbors.__getitem__(random.randint(0, len(neighbors)) - 1)
    for node_index in range(index):
        if batch_memory['Node' + str(node_index + 1)]['id'] == idSelected:
            input_state.append(node_index)
            # print("append node_index -> ", node_index)
            break

    # print('最后看看这里append的size ->', len(input_state))
    # print('len ->', len(input_state), 'input_state ->', input_state)
    # 返回这个idSelected给节点本身同时生成业务号

    # lock.acquire()
    request_id = hashlib.md5((str(random.randint(0, 10000)) + datetime.datetime.now().strftime('%H:%M:%S') + str(
        random.randint(0, 10000))).encode("utf-8")).hexdigest()
    # lock.release()
    # print('request_id ->', request_id)
    data = json.dumps({'request_id': request_id, 'idSelected': idSelected})  # 这里可以直接传id了
    # data = json.dumps({'request_id': request_id, 'idSelected': 'QmcTKvY2y5DjELMTb361ecNNz884jaRMzPRw6eSutDrbWC'})
    res = Response(data, content_type='application/json')

    cache[request_id] = input_state  # 暂时存一下咯
    return res

    # 当开始选择就先放暂时没组织好的数据到cache里面
    # 这一系列都需要server来生成一个业务号码 为什么需要业务号码，因为我们在存储这个状态的时候并非是同步的，需要来回request和return msg ，需要用业务号来标识是哪一个数据集的数据
    # 取出这些邻居的近8s数据展开一下，组织成input data的格式，这里是pop(last) 如何让能采样到8s数据呢？先把go那边的节点启动后都停止个8的倍数个s不就行了...
    # 组织好后，input进model里面，得到model预测的reward

    # 选择最小reward的node id返回 (这里给一个业务号给go那边)
    # 真实的reward在我们收到id后我们就去选择和谁传输数据，然后在go那边统计好reward后返回 (返回业务号和reward序列等数据)
    # next的状态就是我们取出这8s后再从state stack里面取出近8s的数据当作Q（next）输入用来调整神经网络 ,这里不pop只拿last数据出来用
    # 存储这一次的play记录


# 节点上传想要参与选举的节点列表ID
@app.route('/postReward/', methods=['POST'])
def postReward():
    global DuelingNetwork, selectCnt, input_data_lengthn
    jsondata = request.json

    request_id = jsondata['request_id']
    reward = jsondata['reward']

    temp_mem = cache[request_id]
    # print('看看之前的temp_mem ->', len(temp_mem), temp_mem)
    temp_mem.append(reward)
    # print("append reward ->", reward)

    index = len(nodes) - 1
    # batch_memory这里可能需要加个锁
    for node_index in range(index):
        for data_type in batch_memory['Node' + str(node_index + 1)]['datasets']:
            datas = batch_memory['Node' + str(node_index + 1)]['datasets'][data_type]
            for data in datas[-time_duration:]:  # 取尾巴8个数据
                temp_mem.append(data)
    # print('reward infos', jsondata)
    # cache[request_id] = temp_mem
    # print('len ->', len(cache[request_id]), 'cache[request_id] ->', cache[request_id])
    # 至此组装完毕,能当一条数据存进play的memory里面去了
    cache.pop(request_id)
    model_lock.acquire()
    # print('看看temp_mem出问题没 -> ', temp_mem.__getitem__(input_data_length))
    DuelingNetwork.store_transition(temp_mem[:input_data_length],
                                    temp_mem.__getitem__(input_data_length),
                                    temp_mem.__getitem__(input_data_length + 1), temp_mem[-input_data_length:])
    print('select ->', 'Node' + str(temp_mem.__getitem__(input_data_length)),
          batch_memory['Node' + str(int(1 + temp_mem.__getitem__(input_data_length)))]['id'], 'get reward ->',
          temp_mem.__getitem__(input_data_length + 1))
    selectCnt += 1
    if selectCnt >= 4 and selectCnt % 1 == 0:
        DuelingNetwork.learn()
    model_lock.release()
    return "post Reward!"


# db Operation
# def addData():
#     with app.app_context():
#         testdata = RunTimeData(nodeName="testNode1", cpuUtilization=0.1, gpuUtilization=0.1, time='2022/10/13')
#         database.session.add(testdata)
#         database.session.commit()


if __name__ == '__main__':
    # with app.app_context():
    #     db = SQLAlchemy(app)
    #
    #
    #     class RunTimeData(db.Model):  # 这样抽象的数据组织方式也是服了自己了
    #         id = db.Column(db.Integer, primary_key=True)  # id字段，int类型，主键
    #         nodeName = db.Column(db.String(9))
    #         cpuUtilization = db.Column(db.Float)
    #         gpuUtilization = db.Column(db.Float)
    #         time = db.Column(db.String(20))
    #
    #
    #     database = db
    #     database.drop_all()
    #     database.create_all()
    # addData()
    app.run()
