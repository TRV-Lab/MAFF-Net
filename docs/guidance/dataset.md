# 📁 Dataset Preparation: VoD & TJ4DRadSet

This document provides instructions for preparing the **View-of-Delft (VoD)** and **TJ4DRadSet** datasets. Both datasets follow the KITTI-style format.

---

## 📦 View-of-Delft (VoD)

*  Download the official dataset from the [View-of-Delft repository](https://github.com/tudelft-iv/view-of-delft-dataset).

* Create a symbolic link:

```bash
ln -s /your_path/view_of_delft_PUBLIC/ ./data/VoD
```
* Dataset directory structure:
```
View-of-Delft
    ├── lidar                    # Contains LiDAR point clouds
    │   ├── ImageSets
    │   ├── training
    │   │   ├── calib
    │   │   ├── velodyne
    │   │   ├── image_2
    │   │   ├── label_2
    │   └── testing
    │       ├── calib
    │       ├── velodyne
    │       ├── image_2
    │
    ├── radar                    # Radar point clouds (single scan)
    │   ├── ImageSets
    │   ├── training
    │   │   ├── calib
    │   │   ├── velodyne
    │   │   ├── image_2
    │   │   ├── label_2
    │   └── testing
    │       ├── calib
    │       ├── velodyne
    │       ├── image_2
    │
    ├── radar_3frames            # Accumulated radar point clouds (3 frames)
    │   ├── ImageSets
    │   ├── training
    │   │   ├── calib
    │   │   ├── velodyne
    │   │   ├── image_2
    │   │   ├── label_2
    │   └── testing
    │       ├── calib
    │       ├── velodyne
    │       ├── image_2
    │
    └── radar_5frames            # Accumulated radar point clouds (5 frames)
        ├── ImageSets
        ├── training
        │   ├── calib
        │   ├── velodyne
        │   ├── image_2
        │   ├── label_2
        └── testing
            ├── calib
            ├── velodyne
            ├── image_2       
```
* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.vod.vod_dataset_radar create_vod_infos tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

## 📦 TJ4DRadSet

* Download the dataset from the [TJ4DRadSet](https://github.com/TJRadarLab/TJ4DRadSet).
* Create a symbolic link:

```bash
ln -s /your_path/TJ4DRadSet_4DRadar/ ./data/TJ4DRadSet
```

* Dataset directory structure:
```
TJ4DRadSet_4DRadar
    ├── ImageSets
    │   │── train.txt
    |       ...
    │   │── readme.txt
    |
    ├── training
    │   │── calib
    │       ├──000000.txt
    │       ...
    │   │── image_2
    │       ├──000000.png
    │       ...
    │   │── label_2
    │       ├──020000.txt
    │       ...    
    │   │── velodyne
    │       ├──000000.bin
    │       ...  
    ├── Video_Demo
    │   │── seq04.mp4
    │       ...  
```
* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.TJ4DRadSet.TJ4DRadSet_dataset_radar create_TJ_infos tools/cfgs/dataset_configs/TJ4DRadSet_dataset_radar.yaml
```
