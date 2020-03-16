#232. implement Queue using Stacks
class Queue(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.Stack, self.Stack1 = [],[]

    def push(self, x):
        """
        :type x: int
        :rtype: nothing
        """
        self.Stack.append(x)

    def pop(self):
        """
        :rtype: nothing
        """
        self.Stack1.append(self.Stack[0])
        self.Stack.pop()
        return self.Stack1[0]

    def peek(self):
        """
        :rtype: int
        """
        # if not self.outStack:
        #     while self.inStack:
        #         self.outStack.append(self.inStack.pop())
        return self.Stack[0]

    def empty(self):
        """
        :rtype: bool
        """
        return not self.Stack
a = Queue()
a.push(1)
a.push(2)
print(a.peek())
print(a.pop())
print(a.empty())