import threading, traceback, sys

class DL_Thread(threading.Thread):  # The timer class is derived from the class threading.Thread
    def __init__(self, funcName, *args):
        threading.Thread.__init__(self)
        self.args = args
        self.funcName = funcName
        self.exitcode = 0
        self.finished=False
        self.exception = None
        self.exc_traceback = ''

    def run(self):  # Overwrite run() method, put what you want the thread do here
        try:
            self._run()
            self.finished=True
        except Exception as e:
            self.exitcode = 1  # 如果线程异常退出，将该标志位设置为1，正常退出为0
            self.exception = e
            self.exc_traceback = ''.join(traceback.format_exception(*sys.exc_info()))  # 在改成员变量中记录异常信息
            raise self.exception

    def _run(self):
        try:
            self.funcName(*(self.args))
        except Exception as e:
            raise e
