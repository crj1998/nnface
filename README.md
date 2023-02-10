# NNFace
nnface is a pytorch based repo for face classification, detection and segementation.

## Prerequest
### Install
clone and setup

recommend install `gdown` for download file from google drive.

### Datasets
We use [FairFace](https://github.com/joojs/fairface) for classification.

#### FairFace
[FairFace](https://github.com/joojs/fairface) is a face image dataset which is race balanced. It contains 108,501 images from 7 different race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino. Images were collected from the YFCC-100M Flickr dataset and labeled with race, gender, and age groups. 

Images (train&validation set): [Padding=0.25](https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view), [Padding=1.25](https://drive.google.com/file/d/1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL/view).

Labels: [Train](https://drive.google.com/file/d/1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH/view), [validation](https://drive.google.com/file/d/1wOdja-ezstMEp81tX1a-EYkFebev4h7D/view).

The author used dlib's get_face_chip() to crop and align faces with padding = 0.25 in the main experiments (less margin) and padding = 1.25 for the bias measument experiment for commercial APIs.

```bash
# on dataset dir
gdown 1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86
# gdown 1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL
mkdir fairface
unzip -d fairface ./fairface-img-margin025-trainval.zip
cd fairface
gdown 1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH
gdown 1wOdja-ezstMEp81tX1a-EYkFebev4h7D
```

#### CelebAMask-HQ
[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) is a large-scale face image dataset that has 30,000 high-resolution face images selected from the CelebA dataset by following CelebA-HQ. Each image has segmentation mask of facial attributes corresponding to CelebA.

The masks of CelebAMask-HQ were manually-annotated with the size of 512 x 512 and 19 classes including all facial components and accessories such as skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, and cloth.

[Google Drive](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv)


#### WIDER
[WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER dataset. We choose 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images. WIDER FACE dataset is organized based on 61 event classes. For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets. We adopt the same evaluation metric employed in the PASCAL VOC dataset. Similar to MALF and Caltech datasets, we do not release bounding box ground truth for the test images. Users are required to submit final prediction files, which we shall proceed to evaluate.

### Pretrained model


## Training

### Tips
- batch size and learning rate have some inter influence. larger batch size require larger lr.

## Benchmark Evolution
For WIDER face challenage, use evolution.py

## Inference Deploy
### onnx
### tensorRT

## References
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [RetinaFace_Pytorch](https://github.com/supernotman/RetinaFace_Pytorch)
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [两次argsort就能直接拿到样本位次](https://blog.metaron.xyz/get_rank/)