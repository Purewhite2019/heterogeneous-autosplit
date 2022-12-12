from multiprocessing.connection import Connection

import time
import random


def init_server(pipe: Connection):
    """Initializer of the split learning server providing data for the flask server.

    Args:
        pipe (Connection): pipe used for communication between flask and split learning server
    """
    while True:
        if (random.random() < 0.8):
            pipe.send('Accuracy of the current batch')
            pipe.send(random.random())
        else:
            pipe.send('Topology changed to')
            pipe.send(random.randint(0, 10))
        time.sleep(random.random() * 0.5 + 0.5)    # Sleep a random time between 0.5s and 3s.
