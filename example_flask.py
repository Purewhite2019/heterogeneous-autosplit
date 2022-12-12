from multiprocessing.connection import Connection

from flask import Flask
import re


app = Flask('Example')
flask_pipe = None

accs = []
topology = 1


@app.route('/accuracy')
def get_accuracy():
    recv()
    return str(accs)

@app.route('/topology')
def get_topology():
    recv()
    return str(topology)

def recv() -> None:
    while flask_pipe.poll():
        msg = flask_pipe.recv()
        print(msg)
        
        # Different types of messages should be handled here.
        if msg == 'Accuracy of the current batch':
            accs.append(flask_pipe.recv())
        elif msg == 'Topology changed to':
            topology = flask_pipe.recv()
        else:
            raise ValueError(f'Invalid message from the pipe: {msg}')

def init_flask(pipe: Connection) -> None:
    """Initializer of the flask server providing data for the frontend visualization.

    Args:
        pipe (Connection): pipe used for communication between flask and split learning server
    """
    global flask_pipe
    flask_pipe = pipe

    app.run(host='0.0.0.0', port=6000, debug=False)