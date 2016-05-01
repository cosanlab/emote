#Interface to define behavior of a dataset object


class FACRepository:

    def __init__(self, buffer_size=50):
        self.data_buffer = []
        self.buffer_size = buffer_size
        self.load_data_into_buffer(self.buffer_size)

    #Must return tuple (data, label)
    def next_item(self):
        datum = self.data_buffer.pop(0)
        self.load_data_into_buffer(1)
        return datum

    #Must return some number of
    def get_items(self, x):
        list = []

        if len(self.data_buffer) != 0:
            for i in range(x):
                list.append(self.data_buffer.pop(0))

                if len(self.data_buffer) == 0:
                    self.load_data_into_buffer(self.buffer_size)

                #Check again and if still empty, then there's nothing left
                #to load
                if len(self.data_buffer) == 0:
                    break

        return list


    def load_data_into_buffer(self, n):
        if n > self.buffer_size:
            n = self.buffer_size

        self.data_buffer = self.data_buffer + self._load_data(n)

    def _load_data(self, n):
        raise NotImplementedError

    def get_testing_items(self):
        print ("Function not implemented")
