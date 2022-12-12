from multiprocessing.connection import Connection

from flask import Flask


app = Flask('Example')
flask_pipe = None
accs = []

@app.route('/accuracy')
def get_accuracy():
    global flask_pipe
    global accs
    
    accs += flask_pipe.recv()
    
    return str(accs)
    

def init_flask(pipe: Connection):
    """Initializer of the flask server providing data for the frontend visualization.

    Args:
        pipe (Connection): pipe used for communication between flask and split learning server
    """
    global flask_pipe
    flask_pipe = pipe

    app.run(host='0.0.0.0', port=6000, debug=False)