from typing import List, Tuple, Any, Callable, Dict, Union
from mpi4py import MPI

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
    
    def recv(self, non_blocking=False) -> List[Tuple[Tuple, Dict]]:
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
            MPI.COMM_WORLD.isend([(msg, kwmsg)], dest)
        else:
            MPI.COMM_WORLD.send([(msg, kwmsg)], dest)
    
    def recv(self, non_blocking=False) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        it blocks until a message comes.

        Args:
            non_blocking (bool): if set to False, it blocks until a message comes.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        if non_blocking:
            return MPI.COMM_WORLD.recv()
        else:
            return MPI.COMM_WORLD.recv()