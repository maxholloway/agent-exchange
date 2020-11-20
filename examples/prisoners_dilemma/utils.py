class BufferList:
    def __init__(self, maxlen):
        self.data = []
        self.maxlen = maxlen

    def append(self, el):
        if len(self.data) == self.maxlen:
            popped = self.data.pop(0)
        else:
            popped = None
        
        self.data.append(el)
        return popped
    
    def peek(self, i):
        return self.data[-i-1]

    def __len__(self):
        return len(self.data)