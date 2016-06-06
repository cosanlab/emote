class FACRepository:
    """Repositories manage loading data sets and organizing them into FAC objects to be given to bundles.

    All child classes must only implement the _load_data function as child classes will generally only know how to load one data set, while the remaining logic is relatively agnostic

    This class implements a buffer system to only load certain amounts of data at once. If the child class choose to load more data, then that advantage is negated. 

    Child classes implement _load_data() to pass data into the buffer, then models call get_items() or next_item().

    """
    def __init__(self, buffer_size=50):
        """
        :param buffer_size: Size of repository buffer
        :type buffer_size: int
        """
        self.data_buffer = []
        self.buffer_size = buffer_size
        self.load_data_into_buffer(self.buffer_size)

    def next_item(self):
        """ Gets the first item in the buffer

        :returns: a single feature/label datastructure instance
        """
        datum = self.data_buffer.pop(0)
        self.load_data_into_buffer(1)
        return datum

    def get_items(self, x, nonzero=False):
        """ Returns x items from the buffer.

        :param x: number of items to return
        :type x: int
        :param nonzero: If True, will not return labels that are completely zero
        :type nonzero: bool
        
        :returns: list of data of length x
        """
        data = []

        if len(self.data_buffer) != 0:
            while len(data) < x:
                datum = self.data_buffer.pop(0)
                if nonzero and not datum.is_zero():
                    data.append(datum)
                else:
                    data.append(datum)

                if len(self.data_buffer) == 0:
                    self.load_data_into_buffer(self.buffer_size)

                #Check again and if still empty, then there's nothing left
                #to load
                if len(self.data_buffer) == 0:
                    break

        return data

    def get_data_for_au(self, n, au):
        """Gets n items for a specific AU
        
        Different repos might have different ways of handling this so we leave it up to the implementations.

        :raises: NotImplementedError
        """
        raise NotImplementedError

    def reset_repo(self):
        """Resets the repository to as if no data has been returned
            
        :raises: NotImplementedError
        """
        raise NotImplementedError

    def load_data_into_buffer(self, n):
        """Loads n items from repository into the buffer

        :param n: number of items to load
        :type n: int
        """
        if n > self.buffer_size:
            n = self.buffer_size

        self.data_buffer = self.data_buffer + self._load_data(n)

    def _load_data(self, n):
        """ Overridden by child classes to load data from files

        :param n: number of data to load
        :type n: int
        :returns: list of data

        :raises: NotImplementedError
        """
        raise NotImplementedError

    def get_testing_items(self):
        """Gets a full list of data to test a model on. Percent of total data this will be determined by child classes

        :raises: NotImplemented Error
        """
        raise NotImplementedError
