import datetime  # 导入时间和随机数模块
import random

from flask import Flask
from flask import request
from flask import jsonify  # 数据转为json，并以字典的形式传回前端
from flask import render_template
from werkzeug.routing import BaseConverter
from flask_sqlalchemy import SQLAlchemy


class RegexConverter(BaseConverter):
    def __init__(self, url_map, *args):
        super(RegexConverter, self).__init__(url_map)
        self.url = url_map
        self.regex = args[0]  # flask的正则匹配得这么干...


app = Flask(__name__)
app.url_map.converters['re'] = RegexConverter
app.config['SECRET_KEY'] = '123456'  # 密码
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@163.221.29.17:3306/test'  # Tencent Cloud

# db
database = None

# test
nodes = [
    {'name': 'Node1', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node2', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node3', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node4', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node5', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node6', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node7', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node8', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node9', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node10', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
    {'name': 'Node11', 'datasets': {'cpu': 'None', 'gpu': 'None'}},
]


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html', nodes=nodes)


@app.route('/<re(r"Node[0-9]*"):path>')
def details(path):
    # print(path)
    datasets = None
    for node in nodes:
        if node['name'] == path:
            datasets = node['datasets']
    return render_template('details.html', nodeName=path, datasets=datasets)


# 生成实时数据
@app.route('/setData/', methods=['POST'])
def setData():
    json = request.json
    print('recv :', json['node'], json['key'])

    now = datetime.datetime.now().strftime('%H:%M:%S')
    data = {'time': now, 'data': random.randint(1, 10)}
    return jsonify(data)  # 将数据以字典的形式传回


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
