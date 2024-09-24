# Snippet 1: Hello World
print("Hello, World!")

# Snippet 2: Variables and Data Types
# Integer
x = 10

# Float
y = 3.14

# String
name = "Alice"

# Boolean
is_active = True

# List
fruits = ["apple", "banana", "cherry"]

# Dictionary
person = {"name": "Bob", "age": 25}

# Snippet 3: Functions
def greet(name):
    return f"Hello, {name}!"

print(greet("Charlie"))

# Snippet 4: Classes and Objects
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says woof!"

my_dog = Dog("Rex", 5)
print(my_dog.bark())

# Snippet 5: Inheritance
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

my_cat = Cat("Whiskers")
print(my_cat.speak())

# Snippet 6: File I/O
# Writing to a file
with open("sample.txt", "w") as file:
    file.write("This is a sample file.")

# Reading from a file
with open("sample.txt", "r") as file:
    content = file.read()
    print(content)

# Snippet 7: Exception Handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero.")
finally:
    print("Execution complete.")

# Snippet 8: List Comprehensions
squares = [x**2 for x in range(10)]
print(squares)

# Snippet 9: Lambda Functions
add = lambda a, b: a + b
print(add(5, 3))

# Snippet 10: Map, Filter, Reduce
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Map
squared = list(map(lambda x: x**2, numbers))
print(squared)

# Filter
even = list(filter(lambda x: x % 2 == 0, numbers))
print(even)

# Reduce
total = reduce(lambda a, b: a + b, numbers)
print(total)

# Snippet 11: Decorators
def decorator_func(func):
    def wrapper():
        print("Before function call.")
        func()
        print("After function call.")
    return wrapper

@decorator_func
def say_hello():
    print("Hello!")

say_hello()

# Snippet 12: Generators
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for number in countdown(5):
    print(number)

# Snippet 13: Context Managers
from contextlib import contextmanager

@contextmanager
def open_file_cm(name, mode):
    f = open(name, mode)
    try:
        yield f
    finally:
        f.close()

with open_file_cm("another_sample.txt", "w") as f:
    f.write("Writing using context manager.")

# Snippet 14: Regular Expressions
import re

text = "The rain in Spain"
match = re.search(r"ai", text)
if match:
    print("Match found:", match.group())

# Snippet 15: Working with JSON
import json

data = {"name": "Eve", "age": 30, "city": "New York"}
json_str = json.dumps(data)
print(json_str)

parsed = json.loads(json_str)
print(parsed)

# Snippet 16: HTTP Requests
import requests

try:
    response = requests.get("https://api.github.com")
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(f"HTTP Request failed: {e}")

# Snippet 17: Data Classes
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)
print(p)

# Snippet 18: Multithreading
import threading

def print_numbers():
    for i in range(5):
        print(i)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()

# Snippet 19: Multiprocessing
import multiprocessing

def square_number(n):
    return n * n

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(square_number, [1, 2, 3, 4, 5])
    print(results)

# Snippet 20: Asyncio
import asyncio

async def main_asyncio():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# To run asyncio, uncomment the following line:
# asyncio.run(main_asyncio())

# Snippet 21: Web Scraping with BeautifulSoup
from bs4 import BeautifulSoup

url = "https://www.example.com"
try:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    print(soup.title.text)
except Exception as e:
    print(f"Web scraping failed: {e}")

# Snippet 22: Flask Web Application
# Note: Running Flask in the same script will block execution. Uncomment to use.
"""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask!"

if __name__ == "__main__":
    app.run(debug=True)
"""

# Snippet 23: Django Model Example
# Note: Django models require a Django project environment. This is just a standalone example.
"""
# models.py
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
"""

# Snippet 24: Pandas DataFrame
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35]
}
df = pd.DataFrame(data)
print(df)

# Snippet 25: NumPy Arrays
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.mean())

# Snippet 26: Matplotlib Plot
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Sample Plot")
plt.show()

# Snippet 27: Scikit-learn Example
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)
print(model.predict([[6]]))

# Snippet 28: TensorFlow Simple Model
import tensorflow as tf

model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mean_squared_error')
print(model_tf.summary())

# Snippet 29: PyTorch Simple Tensor
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
print(y)

# Snippet 30: SQLAlchemy ORM Example
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

new_user = User(name='Dave')
session.add(new_user)
session.commit()

for user in session.query(User).all():
    print(user.name)

# Snippet 31: Logging Example
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.error("This is an error message.")

# Snippet 32: Unit Testing with unittest
import unittest

def add(a, b):
    return a + b

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

# To run unit tests, uncomment the following lines:
# if __name__ == '__main__':
#     unittest.main()

# Snippet 33: Command Line Arguments with argparse
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', action='store_true',
                        help='sum the integers')
    return parser.parse_args()

# To use argparse, run the script from the command line with arguments.

# Snippet 34: Virtual Environments
# Note: Virtual environment commands are meant to be run in the terminal, not within a Python script.
# Uncomment and run these commands in your shell.
"""
# To create a virtual environment, run the following in the terminal:
python -m venv env

# To activate:
# On Windows:
env\Scripts\activate

# On Unix or MacOS:
source env/bin/activate
"""

# Snippet 35: Packaging with setuptools
# Note: Packaging requires separate setup files and directory structure. This is just an example.
"""
# setup.py
from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'requests',
    ],
)
"""

# Snippet 36: Using Pathlib
from pathlib import Path

p = Path('.')
for file in p.iterdir():
    if file.is_file():
        print(file.name)

# Snippet 37: Handling Dates with datetime
from datetime import datetime, timedelta

now = datetime.now()
print("Current time:", now)

future = now + timedelta(days=10)
print("Future time:", future)

# Snippet 38: Serialization with pickle
import pickle

data_pickle = {'a': 1, 'b': 2}
with open('data.pkl', 'wb') as f:
    pickle.dump(data_pickle, f)

with open('data.pkl', 'rb') as f:
    loaded = pickle.load(f)
    print(loaded)

# Snippet 39: Sending Emails with smtplib
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg.set_content("This is a test email.")
msg['Subject'] = 'Test'
msg['From'] = 'sender@example.com'
msg['To'] = 'receiver@example.com'

# Note: Replace with actual SMTP server details
# Uncomment and configure the following lines to send an email.
"""
with smtplib.SMTP('smtp.example.com', 587) as server:
    server.starttls()
    server.login('user', 'password')
    server.send_message(msg)
"""

# Snippet 40: Working with CSV
import csv

# Writing to CSV
with open('data.csv', 'w', newline='') as file_csv:
    writer = csv.writer(file_csv)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 25])
    writer.writerow(["Bob", 30])

# Reading from CSV
with open('data.csv', 'r') as file_csv_read:
    reader = csv.reader(file_csv_read)
    for row in reader:
        print(row)

# Snippet 41: Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) -1
    while left <= right:
        mid = (left + right) //2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid +1
        else:
            right = mid -1
    return -1

result_bs = binary_search([1,2,3,4,5,6], 4)
print("Binary Search Index:", result_bs)

# Snippet 42: Quick Sort
def quick_sort(arr):
    if len(arr) <=1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x==pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print("Quick Sort:", quick_sort([3,6,8,10,1,2,1]))

# Snippet 43: Fibonacci Sequence
def fibonacci(n):
    a, b = 0,1
    for _ in range(n):
        yield a
        a, b = b, a +b

print("Fibonacci Sequence:")
for num in fibonacci(10):
    print(num)

# Snippet 44: Merge Sort
def merge_sort(arr):
    if len(arr) >1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k =0

        while i < len(L) and j < len(R):
            if L[i]<R[j]:
                arr[k] = L[i]
                i +=1
            else:
                arr[k] = R[j]
                j +=1
            k +=1

        while i < len(L):
            arr[k] = L[i]
            i +=1
            k +=1

        while j < len(R):
            arr[k] = R[j]
            j +=1
            k +=1

arr_merge = [12,11,13,5,6,7]
merge_sort(arr_merge)
print("Merge Sort:", arr_merge)

# Snippet 45: Heap Sort
import heapq

def heap_sort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for _ in range(len(h))]

print("Heap Sort:", heap_sort([3,1,4,1,5,9,2,6,5]))

# Snippet 46: Stack Implementation
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[-1] if self.items else None
    
    def is_empty(self):
        return len(self.items) ==0

stack = Stack()
stack.push(1)
stack.push(2)
print("Stack Pop:", stack.pop())
print("Stack Peek:", stack.peek())

# Snippet 47: Queue Implementation
from collections import deque

class Queue:
    def __init__(self):
        self.queue = deque()
    
    def enqueue(self, item):
        self.queue.append(item)
    
    def dequeue(self):
        return self.queue.popleft() if self.queue else None
    
    def is_empty(self):
        return len(self.queue) ==0

q = Queue()
q.enqueue('a')
q.enqueue('b')
print("Queue Dequeue:", q.dequeue())
print("Is Queue Empty?", q.is_empty())

# Snippet 48: Binary Tree Node
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Snippet 49: Depth-First Search
def dfs(node):
    if node:
        print(node.value)
        dfs(node.left)
        dfs(node.right)

# Snippet 50: Breadth-First Search
def bfs(root):
    queue_bfs = deque([root])
    while queue_bfs:
        node = queue_bfs.popleft()
        print(node.value)
        if node.left:
            queue_bfs.append(node.left)
        if node.right:
            queue_bfs.append(node.right)

# Snippet 51: Linked List Node
class ListNode:
    def __init__(self, value=0, next_node=None):
        self.value = value
        self.next = next_node

# Snippet 52: Reverse Linked List
def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        nxt = current.next
        current.next = prev
        prev = current
        current = nxt
    return prev

# Snippet 53: Detect Cycle in Linked List
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Snippet 54: Binary Search Tree Insert
def bst_insert(root, key):
    if root is None:
        return TreeNode(key)
    if key < root.value:
        root.left = bst_insert(root.left, key)
    else:
        root.right = bst_insert(root.right, key)
    return root

# Snippet 55: In-order Traversal
def inorder_traversal(node):
    if node:
        inorder_traversal(node.left)
        print(node.value)
        inorder_traversal(node.right)

# Snippet 56: Pre-order Traversal
def preorder_traversal(node):
    if node:
        print(node.value)
        preorder_traversal(node.left)
        preorder_traversal(node.right)

# Snippet 57: Post-order Traversal
def postorder_traversal(node):
    if node:
        postorder_traversal(node.left)
        postorder_traversal(node.right)
        print(node.value)

# Snippet 58: Dijkstra's Algorithm
def dijkstra(graph, start):
    queue_dijkstra = []
    heapq.heappush(queue_dijkstra, (0, start))
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] =0
    while queue_dijkstra:
        current_distance, current_vertex = heapq.heappop(queue_dijkstra)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue_dijkstra, (distance, neighbor))
    return distances

graph_dijkstra = {
    'A': {'B':1, 'C':4},
    'B': {'A':1, 'C':2, 'D':5},
    'C': {'A':4, 'B':2, 'D':1},
    'D': {'B':5, 'C':1}
}

print("Dijkstra's Algorithm:", dijkstra(graph_dijkstra, 'A'))

# Snippet 59: Bellman-Ford Algorithm
def bellman_ford(graph_bf, start):
    distance = {v: float('infinity') for v in graph_bf}
    distance[start] =0
    for _ in range(len(graph_bf)-1):
        for u in graph_bf:
            for v, w in graph_bf[u].items():
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
    # Check for negative cycles
    for u in graph_bf:
        for v, w in graph_bf[u].items():
            if distance[u] + w < distance[v]:
                raise ValueError("Graph contains a negative-weight cycle")
    return distance

graph_bellman = {
    'A': {'B':4, 'C':5},
    'B': {'C':-2, 'D':3},
    'C': {'D':2},
    'D': {}
}

print("Bellman-Ford Algorithm:", bellman_ford(graph_bellman, 'A'))

# Snippet 60: K-Means Clustering
from sklearn.cluster import KMeans

X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("K-Means Labels:", kmeans.labels_)
print("K-Means Cluster Centers:", kmeans.cluster_centers_)
