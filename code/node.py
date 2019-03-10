from builtins import range
from functools import reduce
import numpy as np

class Node(object):
    epsilon = 10**(-4)

    def __init__(self, nid):
        self.enabled = True
        self.node_id = nid
        self.neighbours = []
        self.incoming = []
        self.outgoing = []
        self.oldoutgoing = []

    def reset(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True
        for n in self.neighbours:
            n.enabled = True

    def next_step(self):
        """ Used to have this line in prepMessages
            but it didn't work?
        """
        self.oldoutgoing = self.outgoing[:]

    def normalize_messages(self):
        # Normalize to sum to 1
        self.outgoing = [x / np.sum(x) for x in self.outgoing]

    def receive_message(self, f, m):
        if self.enabled:
            i = self.neighbours.index(f)
            self.incoming[i] = m

    def send_messages(self):
        for i in range(0, len(self.outgoing)):
            self.neighbours[i].receive_message(self, self.outgoing[i])

    def check_convergence(self):
        if self.enabled:
            for i in range(0, len(self.outgoing)):
                # check messages have same shape
                self.oldoutgoing[i].shape = self.outgoing[i].shape
                delta = np.absolute(self.outgoing[i] - self.oldoutgoing[i])
                if (delta > Node.epsilon).any(): # if there has been change
                    return False
            return True
        else:
            # Always return True if disabled to avoid interrupting check
            return True

class VariableNode(Node):
    def __init__(self, name, dim, nid):
        super(VariableNode, self).__init__(nid)
        self.name = name
        self.dim = dim
        self.observed = -1 # only >= 0 if variable is observed

    def reset(self):
        super(VariableNode, self).reset()
        size = range(0, len(self.incoming))
        self.incoming = [np.ones((self.dim,1)) for i in size]
        self.outgoing = [np.ones((self.dim,1)) for i in size]
        self.oldoutgoing = [np.ones((self.dim,1)) for i in size]
        self.observed = -1

    def condition(self, observation):
        self.enable()
        self.observed = observation
        # set messages (won't change)
        for i in range(0, len(self.outgoing)):
            self.outgoing[i] = np.zeros((self.dim,1))
            self.outgoing[i][self.observed] = 1.
        self.next_step() # copy into oldoutgoing

    def prepMessages(self):
        # compute new messages if no observation has been made
        if self.enabled and self.observed < 0 and len(self.neighbours) > 1:
            # switch reference for old messages
            self.next_step()
            for i in range(0, len(self.incoming)):
                # multiply together all excluding message at current index
                curr = self.incoming[:]
                del curr[i]
                self.outgoing[i] = reduce(np.multiply, curr)

            # normalize once finished with all messages
            self.normalize_messages()

class FactorNode(Node):
    """ Factor node in factor graph
    """
    def __init__(self, P, nid, *args):
        super(FactorNode, self).__init__(nid)
        self.P = P
        self.neighbours = list(args) # list storing refs to variable nodes

        # num of edges
        neighbours_number = len(self.neighbours)
        dependencies_number = self.P.squeeze().ndim

        # init messages
        for i in range(0, neighbours_number):
            v = self.neighbours[i]
            vdim = v.dim

            # init for factor
            self.incoming.append(np.ones((vdim,1)))
            self.outgoing.append(np.ones((vdim,1)))
            self.oldoutgoing.append(np.ones((vdim,1)))

            # init for variable
            v.neighbours.append(self)
            v.incoming.append(np.ones((vdim,1)))
            v.outgoing.append(np.ones((vdim,1)))
            v.oldoutgoing.append(np.ones((vdim,1)))

        # error check
        assert (neighbours_number == dependencies_number), "Factor dimensions does not match size of domain."

    def reset(self):
        super(FactorNode, self).reset()
        for i in range(0, len(self.incoming)):
            self.incoming[i] = np.ones((self.neighbours[i].dim,1))
            self.outgoing[i] = np.ones((self.neighbours[i].dim,1))
            self.oldoutgoing[i] = np.ones((self.neighbours[i].dim,1))

    def prepMessages(self):
        """ Multiplies incoming messages w/ P to make new outgoing
        """
        if self.enabled:
            # switch references for old messages
            self.next_step()

            mnum = len(self.incoming)

            # do tiling in advance
            # roll axes to match shape of newMessage after
            for i in range(0,mnum):
                # find tiling size
                nextShape = list(self.P.shape)
                del nextShape[i]
                nextShape.insert(0, 1)
                # need to expand incoming message to correct num of dims to tile properly
                prepShape = [1 for x in nextShape]
                prepShape[0] = self.incoming[i].shape[0]
                self.incoming[i].shape = prepShape
                # tile and roll
                self.incoming[i] = np.tile(self.incoming[i], nextShape)
                self.incoming[i] = np.rollaxis(self.incoming[i], 0, i+1)

            # loop over subsets
            for i in range(0, mnum):
                curr = self.incoming[:]
                del curr[i]
                newMessage = reduce(np.multiply, curr, self.P)

                # sum over all vars except i!
                # roll axis i to front then sum over all other axes
                newMessage = np.rollaxis(newMessage, i, 0)
                newMessage = np.sum(newMessage, tuple(range(1,mnum)))
                newMessage.shape = (newMessage.shape[0],1)

                #store new message
                self.outgoing[i] = newMessage

            # normalize once finished with all messages
            self.normalize_messages()
