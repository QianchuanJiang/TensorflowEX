import threading
import time
count = 8
class MyThread(threading.Thread):
    def __init__(self,threadName):
        super(MyThread,self).__init__(name= threadName)

    def run(self):
       global count
       for i in range(100):
           count = count + 1
           time.sleep(0.1)
           print(self.getName(), count)

for i in range(2):
    MyThread("MyThreadName:" + str(i)).start()

