# const
FEATURES_DIM = (512, 7, 7)
EXPECTED_CLASS = 5

if __name__ == '__main__':
    # command line arguments
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    class_file = sys.argv[3]

    print('BATCH_SIZE: {}'.format(BATCH_SIZE))
    print('NB_EPOCH: {}'.format(NB_EPOCH))

    # loading dataset
    print('Loading train dataset: {}'.format(train_file))
    train_datafile = tables.open_file(train_file, mode='r')
    train_dataset = train_datafile.root
    print('Train data: {}'.format((train_dataset.data.nrows,) + train_dataset.data[0].shape))

    print('Loading test dataset: {}'.format(test_file))
    test_datafile = tables.open_file(test_file, mode='r')
    test_dataset = test_datafile.root
    print('Test data: {}'.format((test_dataset.data.nrows,) + test_dataset.data[0].shape))
