# for testing purpose
import unittest
import services.reader_writer as reader_writer


class TestReaderWriter(unittest.TestCase):
    def read_dataset_2(self):
        X,y=reader_writer.baca_data_cus_seg()
        self.assertEqual(X.shape[0],8068,"Should be 8068")
        self.assertEqual(X.shape[1],68,"Should be 68")


if __name__ == "__main__":
    #unittest.main() 
    print("auc"=="auc")