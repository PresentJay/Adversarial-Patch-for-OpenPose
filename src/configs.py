import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers")
    parser.add_argument('--total_num', type=int, default=1000, help="number of dataset images")
    parser.add_argument('--train_size', type=int, default=800, help="number of training images")
    parser.add_argument('--test_size', type=int, default=200, help="number of test images")
    parser.add_argument('--noise_percentage', type=float, default=0.1, help="percentage of the patch size compared with the image size")
    parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
    parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
    parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
    parser.add_argument('--target', type=int, default=859, help="target label")
    parser.add_argument('--epochs', type=int, default=20, help="total epoch")
    parser.add_argument('--data_dir', type=str, default='data/Imagenet1k', help="dir of the dataset")
    parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
    parser.add_argument('--mask_type', type=str, default='rectangle', help="type of the mask")
    parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
    parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
    parser.add_argument('--image_channel', type=int, default=3, help='image color channel')
    parser.add_argument('--image_width', type=int, default=224, help='image width')
    parser.add_argument('--image_height', type=int, default=224, help='image height')
    return parser.parse_args()