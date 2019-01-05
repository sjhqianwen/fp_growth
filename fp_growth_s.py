# encoding: utf-8
import sys
import os
import csv
import operator
from sympy import *
import re
from collections import defaultdict, namedtuple
from itertools import imap
import numpy as np
import matplotlib.pyplot as plt

def get_txList(fname):
    txList = []
    with open(fname, 'r') as fp:
        for line in fp:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            line = line.rstrip(',')
            lst = line.split(',')
            txList.append(lst)
    return txList

def fitting(m,n):
    #对数据进行拟合
    f1 = np.polyfit(m, n, 6)
    print ('1',f1)
    #print (len(f1))
    #print (type(f1))
    #print (f1[0])
    print('f1 is :\n', f1)
    p1 = np.poly1d(f1)
    print ('2', p1)
    print(np.poly1d(f1))
    print (np.polyder(p1,2))
    p2 = np.polyder(p1,2)
    s = solve(p2,x)
    print s
    #s = solve(p2, x)
   # print ('p2 is :\n',p2)
    #print ('3',s[0].evalf())

    # 也可使用yvals=np.polyval(f1, x)
    yvals = p1(m)  # 拟合y值
    print('yvals is :\n', yvals)

    # 绘图
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title('polyfitting')
    plt.show()

if __name__=='__main__':
    if len(sys.argv) == 0:
        print('Please input dataset filename.')
        sys.exit()
    fname = sys.argv[1]
    if not os.path.exists(fname):
        print('%s does not exist.' % (fname) )
    print("fptree:")
    transactions = get_txList(fname)
    items = defaultdict(lambda: 0)
    for transaction in transactions:
        #去掉 空格 项
        for item in transaction:
            if item == '':
                break
            else:
                items[item] += 1
    sorted(items.items(), key=lambda x: x[1], reverse=False)
    sorted_x = sorted(items.items(), key=operator.itemgetter(1),reverse = True)
    a = len(sorted_x)
    x = []
    for i in  range(len(sorted_x)):
        x.append(i)
    y = []
    for item in sorted_x:
        y.append(item[1])
    fitting(x,y)