from typing import List, Tuple, Any, Callable, Dict, Union
from mpi4py import MPI

class Connection():
    """Abstract class defining the connection between 2 endpoints.
    """
    def __init__(self) -> None:
        pass

    def send(self, *msg, **kwmsg) -> None:
        """Send a group of message to message queue of the other endpoint.
        Args:
            msg (Tuple, optional): Message to be sent in Tuple
            kwmsg (Dict, optional): Message to be sent in Dict
            
        Raises:
            NotImplementedError: this is an abstract class that should`n be called.
        """
        raise NotImplementedError()
    
    def recv(self) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        this function keeps waiting until a message comes.
        it blocks until a message comes.

        Raises:
            NotImplementedError: this is an abstract class that should`n be called.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        raise NotImplementedError()


class MPIConnection(Connection):
    """Implementation of Connection using MPI
    """
    def __init__(self, dst_rank) -> None:
        super().__init__()
        self.dst_rank = dst_rank

    def send(self, *msg, **kwmsg) -> None:
        """Send a group of message to message queue of the other endpoint.
        Args:
            msg (Tuple, optional): Message to be sent in Tuple
            kwmsg (Dict, optional): Message to be sent in Dict
        """
        MPI.COMM_WORLD.send([(msg, kwmsg)], self.dst_rank)
    
    def recv(self) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        it blocks until a message comes.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        return MPI.COMM_WORLD.recv(self.dst_rank)