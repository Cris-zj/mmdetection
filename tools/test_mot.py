import argparse
import os

import mmcv
from mmdet.datasets import build_dataloader

from mot_utils import build_dataset, OneShotMOT, TwoStepMOT


def parse_args():
    parser = argparse.ArgumentParser(
        description='detection, reid, deepsort.')
    parser.add_argument('--mot_config', help='mot config file path')
    parser.add_argument('--output_dir', help='output result dir')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = mmcv.Config.fromfile(args.mot_config)

    task_type = cfg.get('task_type', None)
    if task_type == 'one_shot':
        build_tracker = OneShotMOT
    elif task_type == 'two_step':
        build_tracker = TwoStepMOT
    else:
        raise ValueError('Currently we only support two types: '
                         'one_shot and two_step.')
    tracker = build_tracker(cfg.model, **cfg.mot)

    mot_accum_results = []
    total_seqs = sorted(os.listdir(cfg.data.seq_prefix))
    if 'MOT17' in total_seqs[0]:
        seq_names = total_seqs[2::3]
    else:
        seq_names = total_seqs

    num_seqs = len(seq_names)
    print(f'Processing {num_seqs} sequences ...', end='')

    for seq_ind, sequence in enumerate(seq_names):
        print(f'\n[{seq_ind + 1}/{len(seq_names)}] {sequence}')

        img_prefix = os.path.join(
            cfg.data.seq_prefix, sequence, cfg.data.img_prefix)
        ann_file = os.path.join(
            cfg.data.seq_prefix, sequence, cfg.data.ann_file)
        seqinfo_file = os.path.join(
            cfg.data.seq_prefix, sequence, cfg.data.seqinfo_file)
        dataset = build_dataset(img_prefix,
                                ann_file,
                                cfg.data.pipeline,
                                seqinfo_file=seqinfo_file)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=0,
            dist=False,
            shuffle=False
        )
        tracker.init_tracks(frame_rate=dataset.seqinfo['framerate'])
        mot_results = []

        prog_bar = mmcv.ProgressBar(len(data_loader))

        for i, data in enumerate(data_loader):
            # model forward to get detections
            detections = tracker.forward(data, cfg.post_processing)

            # match tracks and detections
            tracker.update(detections)

            # get frame results
            tracks = tracker.output()
            mot_results.append(tracks)

            # evaluate
            gt_ids = data['gt_ids'][0][0].numpy()
            gt_bboxes = data['gt_bboxes'][0][0].numpy()
            gt_labels = data['gt_labels'][0][0].numpy()
            track_ids = tracks['track_ids']
            track_bboxes = tracks['track_bboxes']
            tracker.mot_eval(gt_ids,
                             gt_bboxes,
                             gt_labels,
                             track_ids,
                             track_bboxes,
                             iou_thr=cfg.evaluation.iou_thr,
                             valid_labels=[1])

            prog_bar.update()

        mot_accum_results.append(tracker.mot_accumulation)

        save_file = os.path.join(args.output_dir, f'{sequence}.txt')
        tracker.save_txt(save_file, mot_results)

    _, str_summary = tracker.get_summary(mot_accum_results, seq_names)
    print('\n', str_summary)


if __name__ == "__main__":
    main()
