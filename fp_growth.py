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

def transfer2FrozenDataSet(dataSet):
    frozenDataSet = {}     #创建字典
    #print('1',type(frozenDataSet))
    for elem in dataSet:
        frozenDataSet[frozenset(elem[0])] = elem[1]
    return frozenDataSet


def write(filename,print_data):
    with open(filename, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)
        data = ['频繁项','频次','概率']
        a = []
        for i in data:
            a.append(i.decode("utf-8").encode("gbk"))
        csv_writer.writerow(a)
        for list in print_data:
            #print(list)
            #print list[0]
            #print (len(transactions))
            data = [list[0],list[1],float(list[1])/float(len(transactions))]
            #print data
            csv_writer.writerow(data)
def write1(filename,relation):
    with open(filename, 'wb') as csv_file:
        csv_writer = csv.writer(csv_file)
        data = ['条件模式基', '后缀项', '置信度']
        a = []
        for i in data:
            a.append(i.decode("utf-8").encode("gbk"))
        csv_writer.writerow(a)
        for list in relation:
            #print(list)
            csv_writer.writerow(list)
        #csv_writer.writerow(['置信度：', ])
def minimun(x):
    z = []
    for i in range (0,len(x)):
        c = np.imag(x[i])
        if c == 0:
            d = np.real(x[i])
            z.append(round(d))
    if flag:
        b = min(z)
    else:
        b = max(z)
    print z
    #print (flag,b)
    return b
def fitting(m,n):
    #对数据进行拟合
    #print (m,n)
    if flag:
       f1 = np.polyfit(m, n, 5)  #拟合
    else:
        f1 = np.polyfit(m, n, 5)
    p1 = np.poly1d(f1)
    print(np.poly1d(f1))
    print p1
    p2 = np.polyder(p1,2)   #求二阶导
    s = np.roots(p2)         #求解
    print s
    min = minimun(s)         #取整，取最小值
    print min
    yn1 = p1(min)
    print yn1

#绘图
    yvals = p1(m)  # 拟合y值
    plot1 = plt.plot(m, n, 's', label='original values')
    plot2 = plt.plot(m, yvals, 'r', label='polyfit values')
    #    plot2 = plt.plot(x, yn1, 'g', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.annotate(np.poly1d(f1), xy=(min, yn1), xytext=(0,0),
                 arrowprops=dict(facecolor='black', shrink=0.01),
                 )
    plt.legend(loc=1)  # 指定legend的位置右下角
    plt.title('polyfitting')
    plt.show()
    #print yn1
    return yn1
def min_support(items,k):
    if  type(items) is list:
        sorted_x = sorted(items, key=lambda items: items[k], reverse=True)
        global flag
        flag = 0
    else:
        sorted_x = sorted(items.items(), key=operator.itemgetter(k),reverse = True)
        #global flag
        flag = 1
    x = []
    for i in  range(len(sorted_x)):
        x.append(i)
    #print x  #生成自变量x
    y = []
    for item in sorted_x:
        y.append(item[k])
    #print (k,y)  #生成因变量y
    min1 = fitting(x,y)
    if flag:
        min0 = int((min1))
    else:
        min0 = min1
    return min0

def find_frequent_itemsets(transactions, include_support=False):
    items = defaultdict(lambda: 0) # mapping from items to their supports  创建一个字典，它每一项都是0
    # Load the passed-in transactions and count the support that individual
    # items have.对每一个项进行个数的统计，item为每个项
    for transaction in transactions:
        for item in transaction:
            if item == '':
                break
            else:
                items[item] += 1
    minsup = min_support(items,1)
    #print ("minsup",minsup)
    # Remove infrequent items from the item support dictionary. 剔除不频繁的项集，但只剔除了单个项集，未剔除不频繁的项集组合
    items = dict((item, support) for item, support in items.items()
        if support >= minsup)
    # Build our FP-tree. Before any transactions can be added to the tree, they
    # must be stripped of infrequent items and their surviving items must be
    # sorted in decreasing order of frequency.对一个项集进行筛选过滤，并降序排序

    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        #print('1',transaction)
        transaction.sort(key=lambda v: items[v], reverse=True)
        #print('2',transaction)
        return transaction
#
    master = FPTree()
    for transaction in imap(clean_transaction, transactions):
        master.add(transaction)

#使用后缀发现频繁项
    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minsup and item not in suffix:
                # New winner!
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set

                # Build a conditional tree and recursively search for frequent
                # itemsets within it.
                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set):
                    yield s # pass along the good news to our caller

    # Search for frequent itemsets, and yield the results we find.
    for itemset in find_with_suffix(master, []):
        yield itemset

class FPTree(object):
    """
    An FP tree.

    This object may only store transaction items that are hashable
    (i.e., all items must be valid as dictionary keys or set members).
    """

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        # The root node of the tree.
        self._root = FPNode(self, None, None)

        # A dictionary mapping items to the head and tail of a path of
        # "neighbors" that will hit every node containing that item.
        self._routes = {}

    @property
    def root(self):
        """The root node of the tree."""
        return self._root

    def add(self, transaction):
        """Add a transaction to the tree."""
        point = self._root

        for item in transaction:
            next_point = point.search(item)
            if next_point:
                # There is already a node in this tree for the current
                # transaction item; reuse it.
                next_point.increment()
            else:
                # Create a new point and add it as a child of the point we're
                # currently looking at.
                next_point = FPNode(self, item)
                point.add(next_point)

                # Update the route of nodes that contain this item to include
                # our new node.
                self._update_route(next_point)

            point = next_point

    def _update_route(self, point):
        """Add the given node to the route through all nodes for its item."""
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point # route[1] is the tail
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            # First node for this item; start a new route.
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        """
        Generate one 2-tuples for each item represented in the tree. The first
        element of the tuple is the item itself, and the second element is a
        generator that will yield the nodes in the tree that belong to the item.
        """
        for item in self._routes.iterkeys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        """
        Generate the sequence of nodes that contain the given item.
        """

        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        """Generate the prefix paths that end with the given item."""

        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print ('Tree:')
        self.root.inspect(1)

        print ('Routes:')
        for item, nodes in self.items():
            print ('  %r' % item)
            for node in nodes:
                print ('    %r' % node)

def conditional_tree_from_paths(paths):
    """Build a conditional FP-tree from the given prefix paths."""
    tree = FPTree()
    condition_item = None
    items = set()

    # Import the nodes in the paths into the new tree. Only the counts of the
    # leaf notes matter; the remaining counts will be reconstructed from the
    # leaf counts.
    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                # Add a new node to the tree.
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    # Calculate the counts of the non-leaf nodes.
    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree


class FPNode(object):
    """A node in an FP tree."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        """Add the given FPNode `child` as a child of this node."""

        if not isinstance(child, FPNode):
            raise TypeError("Can only add other FPNodes as children")

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        """
        Check whether this node contains a child node for the given item.
        If so, that node is returned; otherwise, `None` is returned.
        """
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        """The tree in which this node appears."""
        return self._tree

    @property
    def item(self):
        """The item contained in this node."""
        return self._item

    @property
    def count(self):
        """The count associated with this node's item."""
        return self._count

    def increment(self):
        """Increment the count associated with this node's item."""
        if self._count is None:
            raise ValueError("Root nodes have no associated count.")
        self._count += 1

    @property
    def root(self):
        """True if this node is the root of a tree; false if otherwise."""
        return self._item is None and self._count is None

    @property
    def leaf(self):
        """True if this node is a leaf in the tree; false if otherwise."""
        return len(self._children) == 0

    @property
    def parent(self):
        """The node's parent"""
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a parent.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a parent from another tree.")
        self._parent = value

    @property
    def neighbor(self):
        """
        The node's neighbor; the one with the same value that is "to the right"
        of it in the tree.
        """
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("A node must have an FPNode as a neighbor.")
        if value and value.tree is not self.tree:
            raise ValueError("Cannot have a neighbor from another tree.")
        self._neighbor = value

    @property
    def children(self):
        """The nodes that are children of this node."""
        return tuple(self._children.itervalues())

    def inspect(self, depth=0):
        print ('  ' * depth) + repr(self)
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (root)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)

def rulesGenerator(result,rules):
    frequentPatterns = transfer2FrozenDataSet(result)
    for frequentset in frequentPatterns:
        if(len(frequentset) > 1):
            getRules(frequentset,frequentset, rules, frequentPatterns)
def removeStr(set, str):
    tempSet = []
    for elem in set:
        if(elem != str):
            tempSet.append(elem)
    tempFrozenSet = frozenset(tempSet)
    return tempFrozenSet


def getRules(frequentset,currentset, rules, frequentPatterns):
    for frequentElem in currentset:
        subSet = removeStr(currentset, frequentElem)
        confidence = float(frequentPatterns[frequentset]) / float(frequentPatterns[subSet])
        #b = confidence/(float(frequentPatterns[frequentset-subSet])/len(transactions))
        #print ("123",b)
        #print (len(transactions))
        flag = False
        for rule in rules:
            if(rule[0] == subSet and rule[1] == frequentset - subSet):
                flag = True
        if(flag == False):
            rules.append((subSet, frequentset - subSet, confidence))

        if(len(subSet) >= 2):
            getRules(frequentset, subSet, rules, frequentPatterns)

if __name__=='__main__':
    if len(sys.argv) == 0:
        print('Please input dataset filename.')
        sys.exit()
    fname = sys.argv[1]
    if not os.path.exists(fname):
        print('%s does not exist.' % (fname) )
    print("fptree:")
    transactions = get_txList(fname)   #获取数据
    global flag
    result = []
    #find_frequent_itemsets(transactions) 发现频繁项集
    for itemset, support in find_frequent_itemsets(transactions,True):
        result.append((itemset, support))
    rules = []
    #print result
    write('write_test.csv', result)
    rulesGenerator(result,rules)
    #print (rules)
    #print (type(rules))
    p = min_support(rules,2)
    #p = 0.5
    #print p
    res = []
    for one in rules:
        if one[2] > p:
            res.append(one)

    print res
    write1('write1_test.csv',res)
