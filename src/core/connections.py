import sys
from typing import List, Tuple, Any, Callable, Dict, Union
import socket

import pickle
import re


class Connection():
    """Abstract class defining the connection between 2 endpoints.
    """
    def __init__(self) -> None:
        pass

    def send(self, dest, non_blocking=False, *msg, **kwmsg) -> None:
        """Send a group of message to message queue of the other endpoint.
        Args:
            non_blocking (bool): if set to False, it blocks until a message comes.
            msg (Tuple, optional): Message to be sent in Tuple
            kwmsg (Dict, optional): Message to be sent in Dict
            
        Raises:
            NotImplementedError: this is an abstract class that should`n be called.
        """
        raise NotImplementedError()
    
    def recv(self, non_blocking=False, **kwargs) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        it blocks until a message comes.

        Args:
            non_blocking (bool): if set to False, it blocks until a message comes.

        Raises:
            NotImplementedError: this is an abstract class that should`n be called.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        raise NotImplementedError()


from mpi4py import MPI
class MPIConnection(Connection):
    """Implementation of Connection using MPI
    """
    def __init__(self, rank) -> None:
        super().__init__()
        self.rank = rank

    def send(self, dest, non_blocking=False, *msg, **kwmsg) -> None:
        """Send a group of message to message queue of the other endpoint.
        Args:
            non_blocking (bool): if set to False, it blocks until a message comes.
            msg (Tuple, optional): Message to be sent in Tuple
            kwmsg (Dict, optional): Message to be sent in Dict
        """
        if non_blocking:
            MPI.COMM_WORLD.isend([(msg, kwmsg)], dest=dest)
        else:
            MPI.COMM_WORLD.send([(msg, kwmsg)], dest=dest)

    def recv(self, non_blocking=False, **kwargs) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        it blocks until a message comes.

        Args:
            non_blocking (bool): if set to False, it blocks until a message comes.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        if non_blocking:
            return MPI.COMM_WORLD.recv(**kwargs)
        else:
            return MPI.COMM_WORLD.recv(**kwargs)



class TCPConnection(Connection): #59.78.9.42: 50000
    """Implementation of Connection using TCP
    """
    N_SERVER_LISTEN = 128
    BUFFER_SIZE = 1400
    
    def __init__(self, is_server:bool=False, server_ip:str=None, server_port:Union[str, int]=None) -> None:
        """Initialize a TCP Connection

        Args:
            is_server (bool, optional): If this connection is used as a server. Defaults to False.
            server_ip (str, optional): The IP of the server to connect for a client connection.. Defaults to None.
            server_port (Union[str, int], optional): The port of the server. Defaults to None.
        
        Example:
        ```
            server = TCPConnection(is_server=True, server_port=10001)
            client = TCPConnection(is_server=False, server_ip='127.0.0.1', server_port=10001)
        ```
        """
        super().__init__()
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.is_server = is_server
        
        if is_server is True:
            address = ('', int(server_port))
            self.connection.bind(address)
            self.connection.listen(TCPConnection.N_SERVER_LISTEN)
            self.clients = {}
            
        else:
            self.connection.connect((server_ip, server_port))
    
    def try_accept(self) -> None:
        if not self.is_server:
            raise RuntimeError('Client connection shouldn`t call this function.')
        try:
            while True:
                self.connection.setblocking(False)
                socket, addr = self.connection.accept()    # If no client is connection, an exception will be raised.
                print('Connect establish')
                socket.setblocking(True)
                data = socket.recv(TCPConnection.BUFFER_SIZE)
                socket.setblocking(False)
                try:
                    while True:
                        data += socket.recv(TCPConnection.BUFFER_SIZE)
                except:
                    (msg, kwmsg) = pickle.loads(data)
                    assert len(msg) == 1 and re.match(r'Client\d+Ready', msg[0]) and kwmsg == {}
                    client_idx = int(re.sub('Ready', "", re.sub('Client', "", msg[0])))
                    print(f'The client {client_idx} of ({socket}, {addr}) is added into client list')
                    assert client_idx not in self.clients.keys()
                    self.clients[client_idx] = (socket, addr)
        except:
            pass
    
    def send(self, dest, non_blocking=False, *msg, **kwmsg) -> None:
        if self.is_server:
            self.try_accept()
            self.clients[dest][0].setblocking(not non_blocking)
            t = pickle.dumps((msg, kwmsg))
            print(sys.getsizeof(t))
            self.clients[dest][0].send(pickle.dumps((msg, kwmsg)))
        else:
            self.connection.setblocking(not non_blocking)
            self.connection.sendall(pickle.dumps((msg, kwmsg)))

    def recv(self, non_blocking=False, **kwargs) -> List[Tuple[Tuple, Dict]]:
        if self.is_server:
            self.try_accept()
            while len(self.clients.keys()) == 0:
                self.try_accept()
        ret = []
        self.connection.setblocking(False)
        # Server
        if self.is_server:
            while len(ret) == 0:
                for socket, _ in self.clients.values():
                    try:
                        # Get the header part of data
                        data = b''
                        data_new = socket.recv(TCPConnection.BUFFER_SIZE)
                        socket.setblocking(True)
                        # Get the remaining parts of data
                        #try:
                        while True:
                            try:
                                data += data_new
                                (msg, kwmsg) = pickle.loads(data)
                                break
                            except:
                                data_new = socket.recv(TCPConnection.BUFFER_SIZE)
                                continue
                        # while len(data_new) >= 1400:
                        #     data += data_new
                        #     data_new = socket.recv(TCPConnection.BUFFER_SIZE)
                        #     print(len(data_new), len(data))
                        # #except:
                        # data += data_new
                        # print(sys.getsizeof(data))
                        # (msg, kwmsg) = pickle.loads(data)
                        ret.append((msg, kwmsg))
                        socket.setblocking(False)
                    except BlockingIOError:
                        socket.setblocking(False)
                        continue
                if non_blocking:
                    break
        # Client
        else:
            self.connection.setblocking(False)
            while len(ret) == 0:
                try:
                    data = b''
                    data_new = self.connection.recv(TCPConnection.BUFFER_SIZE)
                    self.connection.setblocking(True)
                    # Get the remaining parts of data
                    while True:
                        try:
                            data += data_new
                            (msg, kwmsg) = pickle.loads(data)
                            break
                        except:
                            data_new = self.connection.recv(TCPConnection.BUFFER_SIZE)
                            continue
                    # # try:
                    # while len(data_new) >= 1400:
                    #     data += data_new
                    #     #print(len(data_new), len(data))
                    #     data_new = self.connection.recv(TCPConnection.BUFFER_SIZE)
                    # # except:
                    # data += data_new
                    # # print(sys.getsizeof(data))
                    # (msg, kwmsg) = pickle.loads(data)
                    ret.append((msg, kwmsg))
                    self.connection.setblocking(False)
                except BlockingIOError:
                    self.connection.setblocking(False)
                    continue
                if non_blocking:
                    break
        return ret
