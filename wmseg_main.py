from __future__ import absolute_import, division, print_function

import argparse
import torch

from tqdm import tqdm
from wmseg_eval import eval_sentence
from wmseg_model import WMSeg


def load_model(path, no_cuda=False):
    device, _ = get_device(no_cuda)
    model = torch.load(path, map_location=device)
    seg_model = WMSeg.from_spec(model["spec"], model["state_dict"], device)
    return seg_model


def get_device(no_cuda):
    if no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda")
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    return (device, n_gpu)


def predict(args, seg_model: WMSeg = None):
    if type(args) is dict:
        args = _parse_args(
            [f"--{k}{'' if v is True else '='+v}".strip() for k, v in args.items()]
        )

    device, n_gpu = get_device(args.no_cuda)
    args.device = device.type
    print("device: {} gpu#: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    if seg_model is None:
        model = torch.load(args.model, map_location=device)
        seg_model = WMSeg.from_spec(model["spec"], model["state_dict"], args)

    eval_examples = seg_model.load_data(args.input, do_predict=True)
    convert_examples_to_features = seg_model.convert_examples_to_features
    feature2input = seg_model.feature2input
    num_labels = seg_model.num_labels
    word2id = seg_model.word2id
    label_map = {v: k for k, v in seg_model.labelmap.items()}

    if args.fp16:
        seg_model.half()
    seg_model.to(device)
    if n_gpu > 1:
        seg_model = torch.nn.DataParallel(seg_model)

    seg_model.to(device)

    seg_model.eval()
    y_pred = []

    for start_index in tqdm(range(0, len(eval_examples), args.batch)):
        eval_batch_examples = eval_examples[
            start_index : min(start_index + args.batch, len(eval_examples))
        ]
        eval_features = convert_examples_to_features(eval_batch_examples)

        (
            input_ids,
            input_mask,
            l_mask,
            label_ids,
            matching_matrix,
            ngram_ids,
            ngram_positions,
            segment_ids,
            valid_ids,
            word_ids,
            word_mask,
        ) = feature2input(device, eval_features)

        with torch.no_grad():
            _, tag_seq = seg_model(
                input_ids,
                segment_ids,
                input_mask,
                labels=label_ids,
                valid_ids=valid_ids,
                attention_mask_label=l_mask,
                word_seq=word_ids,
                label_value_matrix=matching_matrix,
                word_mask=word_mask,
                input_ngram_ids=ngram_ids,
                ngram_position_matrix=ngram_positions,
            )

        logits = tag_seq.to("cpu").numpy()
        label_ids = label_ids.to("cpu").numpy()

        for i, label in enumerate(label_ids):
            temp = []
            for j, _ in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == num_labels - 1:
                    y_pred.append(temp)
                    break
                else:
                    temp.append(label_map[logits[i][j]])

    print("write results to %s" % str(args.output))
    with open(args.output, "w", encoding="utf8") as writer:
        for i in range(len(y_pred)):
            sentence = eval_examples[i].text_a
            _, seg_pred_str = eval_sentence(y_pred[i], None, sentence, word2id)
            writer.write("%s\n" % seg_pred_str)


def _parse_args(v=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="The data path containing the sentences to be segmented",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="The output path of segmented file",
    )
    parser.add_argument("-m", "--model", default=None, type=str, help="")
    parser.add_argument(
        "-b", "--batch", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )

    args = parser.parse_args(v)

    if args.output is None and args.input is not None:
        args.output = args.input + ".out"

    return args


def main():
    predict(_parse_args())


if __name__ == "__main__":
    main()
