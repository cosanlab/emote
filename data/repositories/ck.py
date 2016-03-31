#Interface to access CK+ data set within python
import cv2

from data.FAC import FACDatum, FACLabel
from data.repositories.FACRepository import FACRepository
from util.paths import get_real_path

DATA_DIR = 'ck_cleaned'
FAC_DATA = 'fac_data'
BUFFER_SIZE = 50
IMAGE_EXT = '.jpg'

class CKRepository(FACRepository):

    def __init__(self):
        self.data_dir = get_real_path(__file__) + '/../' + DATA_DIR
        self.fac_data = self.data_dir + '/' + FAC_DATA
        self.current = 1

        FACRepository.__init__(self, BUFFER_SIZE)


    def _load_data(self, n):
        fp = open(self.fac_data)
        data = []

        for i, line in enumerate(fp):
            if i >= self.current and i < self.current + n:
                data.append(self._line_to_datum(line))

        self.current += n
        fp.close()

        return data

    def _line_to_datum(self, line):

        data = line.strip().split(" ")
        labels = []
        for i in range(1, len(data), 2):
            label = FACLabel(float(data[i]), float(data[i+1]))
            labels.append(label)

        image = cv2.imread(self.data_dir + '/' + data[0] + IMAGE_EXT)
        datum = FACDatum(image, labels, data[0])

        return datum


def main():
    ck = CKRepository()

    items = ck.get_items(20)
    items += ck.get_items(50)
    items.append(ck.next_item())
    items.append(ck.next_item())

    uni = set(items)

    print("Buffer Size: " + str(len(ck.data_buffer)))
    print("Item length: " + str(len(items)))
    print("Set size: " + str(len(uni)))

if __name__ == '__main__':
    main()
