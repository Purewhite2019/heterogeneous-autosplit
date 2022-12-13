import multiprocessing

from example_server import init_server
from example_flask import init_flask


if __name__ == '__main__':
    pipe = multiprocessing.Pipe()
    proc_flask = multiprocessing.Process(target=init_flask, args=(pipe[0], ))
    proc_server = multiprocessing.Process(target=init_server, args=(pipe[1], ))
    
    proc_flask.start()
    proc_server.start()
    
    proc_flask.join()
    proc_server.join()