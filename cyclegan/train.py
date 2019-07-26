import data_io


def train(batch_size, epoch_count, dataset_folder_path, output_path):
    dataset = data_io.dataset_load(dataset_folder_path)
    train_iter = SerialIterator(dataset, batch_size, repeat=True, shuffle=True)


if __name__ == '__main__':
    now_time = datetime.datetime.now()
    folder_name = "{0:%Y-%m-%d_%H-%M-%S}".format(now_time)

    parser = argparse.ArgumentParser(description='Chainer-GAN: cyclegan')
    parser.add_argument('--batchsize', '-b', type=int, default=55,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-d', default='dataset',
                        help='Directory of dataset image files. Default is /Chainer-GAN/cyclegan/dataset')
    parser.add_argument('--out', '-o', default='output/' + folder_name + "/",
                        help='Directory to output the result. Default is /Chainer-GAN/cyclegan/output/yyyy-mm-dd_HH-MM-SS')
    args = parser.parse_args()

    train(args.batchsize, args.epoch, args.dataset, args.out)
