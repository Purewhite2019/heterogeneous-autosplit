from multiprocessing.connection import Connection

import time
import random


def init_server(pipe: Connection):
    """Initializer of the split learning server providing data for the flask server.

    Args:
        pipe (Connection): pipe used for communication between flask and split learning server
    """
    while True:
        pipe.send(f'Accuracy of current batch: {random.random()}')  # Send the accuracy of the current batch to the flask server.
        time.sleep(random.random() * 2.5 + 0.5)    # Sleep a random time between 0.5s and 3s.
