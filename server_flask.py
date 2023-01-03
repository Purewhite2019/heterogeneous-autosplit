from flask import Flask
from flask_cors import CORS

import re
import logging
import json
from src.utils.utils import AverageMeter
from multiprocessing.connection import Connection


app = Flask('heterogeneous-autosplit-flask')
CORS(app)
flask_pipe = None

accuracy_meters = dict()
client_infos = dict()


@app.route('/data')
def get_data() -> str:
    update()
    return json.dumps(client_infos)

def update() -> None:
    global client_infos
    global topology
    
    while flask_pipe.poll():
        client_idx, layer_idx, corrects, n_samples, short_summary, feat_size = flask_pipe.recv()
        logging.debug(f'Received information of client {client_idx}')
        
        if client_idx not in client_infos.keys():
            accuracy_meters[client_idx] = AverageMeter()
            client_infos[client_idx] = {
                'accuracy': [],
            }
        accuracy_meters[client_idx].update(corrects / n_samples)
        client_infos[client_idx]['accuracy'].append(accuracy_meters[client_idx].exp_avg)
        client_infos[client_idx]['feat_size'] = feat_size
        client_infos[client_idx]['time'] = short_summary
        client_infos[client_idx]['n_layer'] = layer_idx

def init_flask(pipe: Connection) -> None:
    """Initializer of the flask server providing data for the frontend visualization.

    Args:
        pipe (Connection): pipe used for communication between flask and split learning server
    """
    global flask_pipe
    flask_pipe = pipe

    app.run(host="127.0.0.1", port=6000)
