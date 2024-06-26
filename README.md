# Fine-grained Distribution Alignment via Double-Check for Robust Semi-supervised Object Detection

![](./resources/pipeline.png)




## Main Results

### Partial Labeled Data

The results are shown in the following:

#### 
| Method | 1% | 2% | 5% |10%|100%|
| ---- | -------| ---- | ----- |----|----|
| Ours | 24.72 | 28.42 | 34.10 | 37.30 | 46.20 |

#### 
## Usage

### Requirements
- `Ubuntu 16.04`
- `Anaconda3` with `python=3`
- `Pytorch`
- `mmdetection`
- `mmcv`

#### Notes
- Our codes are modified from [E2E Soft Teacher](https://github.com/microsoft/SoftTeacher)
- The project is based on `mmdetection v2.16.0`.
### Installation
```
make install
```

### Data Preparation
- Download the COCO dataset
- Execute the following command to generate data set splits:

```shell script
# YOUR_DATA should be a directory contains coco dataset.
# For eg.:
# YOUR_DATA/
#  coco/
#     train2017/
#     val2017/
#     unlabeled2017/
#     annotations/
ln -s ${YOUR_DATA} data
bash tools/dataset/prepare_coco_data.sh conduct
```

### Training
- To train model on the **partial labeled data** setting:
```shell script
# JOB_TYPE: 'baseline' or 'semi', decide which kind of job to run
# PERCENT_LABELED_DATA: 1, 5, 10. The ratio of labeled coco data in whole training dataset.
# GPU_NUM: number of gpus to run the job
for FOLD in 1 2 3 4 5;
do
  bash tools/dist_train_partially.sh <JOB_TYPE> ${FOLD} <PERCENT_LABELED_DATA> <GPU_NUM>
done
```
For example, we could run the following scripts to train our model on 10% labeled data with 4 GPUs:

```shell script
bash tools/dist_train_partially.sh semi ${FOLD} 10 4
# ${FOLD} can be 1 2 3 4 5
```

- To train model on the **full labeled data** setting:

```shell script
bash tools/dist_train.sh <CONFIG_FILE_PATH> <NUM_GPUS>
```
For example, to train ours `R50` model with 8 GPUs:
```shell script
bash tools/dist_train.sh configs/soft_teacher/DCST_faster_rcnn_r50_caffe_fpn_coco_full_720k.py 8
```
- To train model on **new dataset**:

The core idea is to convert a new dataset to coco format. Details about it can be found in the [adding new dataset](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/customize_dataset.md).



### Evaluation
```
bash tools/dist_test.sh <CONFIG_FILE_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval bbox --cfg-options model.test_cfg.rcnn.score_thr=<THR>
```

