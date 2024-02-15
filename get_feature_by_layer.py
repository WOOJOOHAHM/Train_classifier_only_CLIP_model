import argparse
import sys
sys.path.append('/hahmwj/Cloned_model/CLIP/')
import clip

import torch
from tqdm import tqdm
from PIL import Image
from util import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data name.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per gpu')
    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_train, dataloader_val, dataloader_test, num_classes = load_data(args.data, '/hahmwj/csv_files/', num_workers=0)
    model, preprocess = clip.load("ViT-B/16", device=device, num_classes = num_classes)

    for name, param in model.named_parameters():
        if name != 'fc.weight' and name != 'fc.bias':  # fc 레이어의 파라미터는 제외
            param.requires_grad = False
        else:
            print(f' Learnable Parameter {name}')
            param.requires_grad = True


    optimizer = torch.optim.SGD(params = model.fc.parameters(), lr = 1e-4, momentum = 0.9, weight_decay=1e-2)
    model = train(model, dataloader_train, dataloader_val, 50, optimizer, args.data)
    _, _, _ = evaluate(model, dataloader_val, True, args.data)

    # for data, label in tqdm(dataloader_test):
    #     data, label = data.to(device), label.to(device)
    #     with torch.no_grad():
    #         output, att1, att2, att3, att4, att5 = model(data)
    #         video_differs, att1_differs, att2_differs, att3_differs, att4_differs, att5_differs = get_difference(output, att1, att2, att3, att4, att5)
    #         att1 = att1.permute(1, 0, 2)
    #         data = data.flatten(0, 1).permute(1, 0, 2, 3).flatten(2, 3).flatten(1, 2)
    #         print(data.shape[1])
    #         print(sum(abs(data[0] - data[1])/max(abs(data[0] - data[1]))/data.shape[1]))
    #         print(att1.size())
    #         print(asd)
    # print(output[0].size())
    



if __name__ == '__main__': main()