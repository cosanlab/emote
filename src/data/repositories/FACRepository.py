class FACRepository:
    """Repositories manage loading data sets and organizing them into FAC objects to be given to bundles.

    All child classes must only implement the _load_data function as child classes will generally only know how to load one data set, while the remaining logic is relatively agnostic

    This class implements a buffer system to only load certain amounts of data at once. If the child class choose to load more data, then that advantage is negated. 

    Child classes implement _load_data() to pass data into the buffer, then models call get_items() or next_item().

    """
    def __init__(self, batch_size):
        self.batch_size = batch_size


    def get_training_batch(self, batch_size):
        """Get next batch of training examples

        :param batch_size: (int) size of the batch to be returned
        :return: list of training examples and features
        """
        raise NotImplementedError

    def get_validation_data(self):
        """List of validation data

        :raises: NotImplementedError
        :return: Validation data
        """
        raise NotImplementedError

    def get_testing_data(self):
        """Gets a full list of data to test a model on. Percent of total data this will be determined by child classes
        :returns: List of testing data
        :raises: NotImplementedError
        """
        raise NotImplementedError
