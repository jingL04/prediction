from torch.utils.data.dataloader import DataLoader
from tool.imblearn.over_sampling import SMOTE
import logging
import argparse
import torch.utils.data as Data
import datetime
from utils.Tools import *
from utils.losses import SupConLoss

parser = argparse.ArgumentParser(description="AGCH demo")
parser.add_argument('--bits', default=100, type=str, help='binary code length (default: 100)')
parser.add_argument('--gpu', default='0', type=str, help='selected gpu (default: 0)')
parser.add_argument('--batch-size', default=2048, type=int, help='batch size (default: 2048)')
parser.add_argument('--NUM-EPOCH', default=30, type=int, help='hyper-parameter: EPOCH (default: 30)')
parser.add_argument('--LOOP', default=10, type=int, help='hyper-parameter: LOOP (default: 10)')
parser.add_argument('--LAMBDA1', default=1, type=float, help='hyper-parameter:  (default: 10**1)')
parser.add_argument('--LAMBDA2', default=0.6, type=float, help='hyper-parameter:  (default: 10**1)')
parser.add_argument('--LAMBDA3', default=1, type=float, help='hyper-parameter:  (default: 10**1)')
parser.add_argument('--LR-A', default=0.00005, type=float, help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-D', default=0.01, type=float, help='hyper-parameter: learning rate (default: 10**-2)')
parser.add_argument('--MOMENTUM', default=0.8, type=float, help='hyper-parameter: momentum (default: 0.8)')
parser.add_argument('--WEIGHT-DECAY', default=0.005, type=float, help='hyper-parameter: weight decay (default: 10**-3)')
parser.add_argument('--NUM-WORKERS', default=0, type=int, help='workers (default: 1)')


class Session:
    def __init__(self):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # Adjustable parameter
        REGENERATE = False
        data_path = 'data/data-balance/'
        IMBALANCE_PROCESSOR = SMOTE()

        root_path_source = 'data/projects/'
        root_path_csv = 'data/csvs/'
        package_heads = ['org', 'gnu', 'bsh', 'javax', 'com', 'fr']

        # Start time
        start_time = datetime.datetime.now()
        start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

        # Get a list of source and target projects
        path_train_and_test = []
        with open('data/pairs-one.txt', 'r') as file_obj:
            for line in file_obj.readlines():
                line = line.strip('\n')
                line = line.strip(' ')
                path_train_and_test.append(line.split(','))

        # Loop each pair of combinations
        for path in path_train_and_test:
            # Get file
            path_train_source = root_path_source + path[0]
            path_train_handcraft = root_path_csv + path[0] + '.csv'
            path_test_source = root_path_source + path[1]
            path_test_handcraft = root_path_csv + path[1] + '.csv'

            # Regenerate token or get from dump_data
            print(path[0] + "===" + path[1])
            train_project_name = path_train_source.split('/')[2]
            test_project_name = path_test_source.split('/')[2]
            path_train_and_test_set = data_path + train_project_name + '_to_' + test_project_name
            # If you don't need to regenerate, get it directly from dump_data
            if os.path.exists(path_train_and_test_set) and not REGENERATE:
                obj = load_data(path_train_and_test_set)
                [train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
            else:
                # Get a list of instances of the training and test sets
                train_file_instances = extract_handcraft_instances(path_train_handcraft)
                test_file_instances = extract_handcraft_instances(path_test_handcraft)

                # Get tokens
                dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
                dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

                # Turn tokens into numbers
                list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
                dict_encoding_train = list_dict[0]
                dict_encoding_test = list_dict[1]

                # Take out data that can be used for training
                train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
                test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

                # Imbalanced processing
                train_ast, train_hand_craft, train_label = imbalance_process(train_ast, train_hand_craft, train_label,
                                                                             IMBALANCE_PROCESSOR)

                # Saved to dump_data
                obj = [train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len,
                       vocabulary_size]
                dump_data(path_train_and_test_set, obj)

            # train
            self.train_ast = torch.tensor(train_ast)
            self.train_hand = torch.tensor(train_hand_craft)
            self.train_label = torch.tensor(train_label)
            self.test_ast = torch.tensor(test_ast)
            self.test_hand = torch.tensor(test_hand_craft)
            self.test_label = torch.tensor(test_label)
            self.train_data = Data.TensorDataset(self.train_ast, self.train_hand, self.train_label)
            # test
            self.test_data = Data.TensorDataset(self.test_ast, self.test_hand, self.test_label)

            # Data Loader (Input Pipeline)
            self.train_loader = DataLoader(dataset=self.train_data,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.NUM_WORKERS,
                                           drop_last=False)

            self.test_loader = DataLoader(dataset=self.test_data,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.NUM_WORKERS,
                                          drop_last=False)
            self.path1 = path[0]
            self.path2 = path[1]

    def define_model(self, coed_length):

        self.CodeNet_A = ASTNet(code_len=coed_length)
        self.gcn_L = GCNL(code_len=coed_length)
        self.CodeNet_H = HANDNet(code_len=coed_length)
        self.Discriminator = discriminator(code_len=coed_length)
        self.Classfication = classfication(code_len=coed_length)
        self.decoderA = decoder_ast(code_len=coed_length)
        self.decoderH = decoder_hand(code_len=coed_length)
        self.attentionA = Attention_ast(code_len=coed_length)
        self.attentionB = Attention_hand(code_len=coed_length)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.criterion = SupConLoss(temperature=0.07)

        self.opt_A = torch.optim.SGD(list(self.CodeNet_A.parameters())+list(self.gcn_L.parameters())+list(self.CodeNet_H.parameters()) +
                                     list(self.decoderA.parameters())+list(self.decoderH.parameters()) + list(self.Classfication.parameters()) +
                                     list(self.attentionA.parameters())+list(self.attentionB.parameters()), lr=args.LR_A,
                                    momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)


        self.opt_D = torch.optim.SGD(self.Discriminator.parameters(), lr=args.LR_D, momentum=args.MOMENTUM,
                                     weight_decay=args.WEIGHT_DECAY)


    def train(self, epoch, args,path):
        self.CodeNet_A.cuda().train()
        self.CodeNet_H.cuda().train()
        self.gcn_L.cuda().train()
        self.Discriminator.cuda().train()
        self.Classfication.cuda().train()
        self.decoderA.cuda().train()
        self.decoderH.cuda().train()
        self.attentionA.cuda().train()
        self.attentionB.cuda().train()

        for idx, (train_ast, train_hand, train_label) in enumerate(self.train_loader):
            train_ast = Variable(train_ast.cuda())
            train_hand = Variable(torch.FloatTensor(train_hand.numpy()).cuda())
            train_label = Variable(train_label.cuda())
            a = train_ast.shape[0]

            self.opt_A.zero_grad()
            self.opt_D.zero_grad()

            F_A1, hid_A, code_A = self.CodeNet_A(train_ast)
            F_H1, hid_H, code_H = self.CodeNet_H(train_hand)
            D_A = self.decoderA(hid_A)
            D_H = self.decoderH(hid_H)
            F_J = torch.cat((hid_A, hid_H), 0)
            label_J = torch.cat((train_label, train_label), 0)

            adj = compute_adj(label_J.float(), path)
            in_aff, out_aff = normalize(adj.type(torch.FloatTensor))
            code_J = self.gcn_L(F_J, in_aff, out_aff)


            att_A1 = self.attentionA(code_A)
            att_A2 = self.attentionA(code_J[0:a,:])
            att_A = torch.cat((att_A1, att_A2), 1)
            att_A = torch.softmax(att_A, dim=1)
            att_A1 = att_A[:,:1]
            att_A2 = att_A[:,1:]
            F_A = torch.multiply(att_A1, code_A) + torch.multiply(att_A2, code_J[0:a,:])

            att_H1 = self.attentionB(code_H)
            att_H2 = self.attentionB(code_J[a:, :])
            att_H = torch.cat((att_H1, att_H2), 1)
            att_H = torch.softmax(att_H, dim=1)
            att_H1 = att_H[:,:1]
            att_H2 = att_H[:,1:]
            F_H = torch.multiply(att_H1, code_H) + torch.multiply(att_H2, code_J[a:,:])
            F_J_0 = torch.concat((F_A, F_H), 1)

            label_1 = train_label.squeeze()
            features = torch.cat([F_A.unsqueeze(1), F_H.unsqueeze(1)], dim=1)
            loss3 = self.criterion(features, label_1)

            B_A_0 = self.Discriminator(F_A)
            B_H_0 = self.Discriminator(F_H)
            all_B_A_0 = torch.ones(a)
            all_B_H_0 = torch.zeros(a)
            all_B_A_0 = all_B_A_0.long().cuda()
            all_B_H_0 = all_B_H_0.long().cuda()
            loss1 = self.CrossEntropyLoss(B_A_0, all_B_A_0) + self.CrossEntropyLoss(B_H_0, all_B_H_0)

            train_labels = torch.squeeze(train_label.long())
            train_labels = train_labels.long().cuda()
            B_J_0 = self.Classfication(F_J_0)
            loss2 = self.CrossEntropyLoss(B_J_0, train_labels.long())


            loss = args.LAMBDA2 * loss2 + args.LAMBDA3 * loss3 - loss1
            loss5 = loss1

            loss.backward(retain_graph=True)
            loss5.backward()

            self.opt_A.step()
            self.opt_D.step()

            print('Epoch [%d/%d], Loss: %.4f'% (epoch + 1, args.NUM_EPOCH, loss.item()))


def main():
    global logdir, args

    args = parser.parse_args()
    sess = Session()
    path_1 = sess.path1
    path_2 = sess.path2

    for loop in range(args.LOOP):
        sess.define_model(args.bits)
        print('--------------------------train Stage--------------------------')
        print('LOOP [%d/%d]' % (loop + 1, args.LOOP))
        print('----------------------------------------------------')
        for epoch in range(args.NUM_EPOCH):
            sess.train(epoch, args, path_1)


if __name__ == "__main__":
    main()
