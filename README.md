# Documentation for Particul: Part Identification using fast-Converging Unsupervised Learning

Particul is an unsupervised algorithm for learning recurring patterns from a database of inputs with low variance.
Inputs are first processed through a feature extractor called a backbone (in the case of images, it is usually a deep convolutional neural network - CNN - pretrained on the ImageNet database) to produce a feature map. Each pattern detector is implemented as a convolution applied on the feature map to produce an activation map. During training, objective functions are implemented in order to ensure that:
* Each detector focus on one region of the activation map (locality);
* All detectors focus on different regions of the activation map (unicity);
* All detectors focus on adjacent regions of the activation map (clustering).

## Fetch dependencies
For installing all required packages.
```bash
cd pytorch_toolbox
python3 -m pip install -e . -r requirements.txt
cd ../particul
python3 -m pip install -r requirements.txt
```

## Learning pattern detectors
### Building and training a model
We start by building a model for learning 6 pattern detectors using the VGG19 backbone
```bash
mkdir models
./tools/detector/build.py -m models/untrained_model.pth \
    -v --particul-backbone vgg19 --particul-npatterns 6
```
Then, we train our model using the CUB200 dataset with the following parameters
* Use RMSprop optimizer with initial learning 9e-4 and decay 1e-4
* Train for 30 epochs while saving a checkpoint every 5 epochs in "checkpoint.tar" (allows to resume computation with --checkpoint-from \<file\> option)
* Use an Unicity ratio of 0.8
* Preprocessing function used on the training set: Random cropping to 448x448 followed by random horizontal flip
* Preprocessing function used on the test set: Center cropping to 448x448
* Normalization: Imagenet
```bash
./tools/detector/train.py \
	-m models/untrained_model.pth -o models/trained_model.pth \
	--verbose \
	--dataset-name CUB200 --dataset-location ./data --dataloader-shuffle --batch-size 16 --seed 0 \
	--preprocessing RandCrop448+ToTensor CenterCrop448+ToTensor --normalization imagenet \
	--optimizer RMSprop lr=1e-4 decay=1e-5 --epochs 30 \
	--checkpoint-every 5 --checkpoint-to checkpoint.tar \
	--particul-loss-unicity 0.8 \
	--device cpu
```
It is also possible to change the target device to cuda:0 if CUDA support is available.

### Calibration
After training, the optional confidence measure can be calibrated on the training set, preferably using a deterministic
preprocessing.
```bash
./tools/detector/calibrate.py \
	-m models/trained_model.pth -o models/calibrated_model.pth \
	--verbose \
	--dataset-name CUB200 --dataset-location ./data --batch-size 16 \
	--preprocessing CenterCrop448+ToTensor --normalization imagenet \
	--device cpu
	--plot
```
Note: The `--plot` option computes confidence statistics on the test set that are displayed along
with the logistic distribution infered on the training set.

![Calibration](/docs/images/calibration.png "Calibration")

### Visualizing results
The following visualization methods are currently implemented:
* upsampling: Simple bilinear upsampling of the activation map of each pattern detector
* saliency: Build saliency map using single-pass gradient computation.
* smooth_grads: Build saliency map using 10 passes of Smooth Grads.
* integrated_gradients: Build saliency map using 2 runs of integrated gradients with random baseline and 50 steps.
```bash
./tools/detector/view.py \
	-m models/calibrated_model.pth \
	--dataset-name CUB200 --dataset-location ./data --dataloader-shuffle --batch-size 16 --seed 0 \
	--preprocessing CenterCrop448+ToTensor --normalization imagenet \
	--nimages 5 \
	--methods smooth_grads \
	--post mask
	--device cuda:0
```
![Visualization methods: from upsampling to integrated gradients](/docs/images/view_methods.png "Visualization methods: upsampling (left), saliency (center), integrated gradients (right)")

Each importance map is normalized between 0 and 1, and can be visualized either as:
* a heatmap (--post jet)
* a heatmap on black background (--post jet-black)
* a binary mask retaining only regions of high importance (--post mask)
* a cropped image corresponding to the region of highest importance (--post extract)
* an aggregation of all parts masks (--post aggregate)

![Post-processing examples](/docs/images/view_post_processing.png "Post-processing options: jet (left), mask (center), extract (right)")


After calibration, the pattern visualization tool now also displays the confidence measure for each prediction.

![Visualization after calibration](/docs/images/view_calibrated.png "Visualization after calibration")


## Out-of-Distribution detection
The confidence measure associated with each detector can also be applied to the detection of Out-of-Distribution (OoD)
inputs (i.e. inputs that differ from images present in the training set).
* In the context of fine-grained recognition, detectors are trained globally (regardless of the category of each image
in the training set). This is the ideal case since Particul detectors are trained using the entire dataset. In this
case, we simply build, train and calibrate detectors using the procedure described [here](#learning-pattern-detectors).
* Otherwise, we attach our detectors to an existing classifier.

### Plugging a Particul-based OoD detector (Particul-OD2) to an existing model
We build a Particul-OD2 by specifying:
* the classifier and name of the layer on which it is attached (here, the detector is attached to the last convolutional
layer of a Resnet50 trained on Caltech101)
* the number of pattern detectors per class
* the shape of the classifier input (this helps to determine the size of the internal feature map on which detectors
are attached)

```bash
./tools/od2/build.py -m models/od2_module_untrained.pth -v \
   --od2-classifier models/resnet_caltech101.pth --od2-layer layer4 \
   --od2-npatterns 6  --od2-ishape 3 32 32
```
### Training a Particul-OD2 detector
Detectors are learned and calibrated using a training set with labels.
```bash
./tools/od2/train.py \
   -m models/od2_module_untrained.pth -o models/od2_module_trained.pth \
   --od2-classifier models/resnet_caltech101.pth \
   --preprocessing Resize224+ToTensor --normalization imagenet \
   --dataset-name Caltech101 --dataset-location ./data \
   --dataloader-shuffle --batch-size 16 \
   --epochs 15 --optimizer RMSprop lr=1e-4 decay=1e-5 \
   --particul-loss-unicity 1.0 --particul-loss-clustering 0.0 \
   --device cuda:0 --seed 0 -v
```
```bash
./tools/od2/calibrate.py \
   --od2-classifier models/resnet_caltech101.pth \
   --od2-detector models/od2_module_trained.pth \
   -o models/od2_module_calibrated.pth --use-labels \
   --device cuda:0 \
   --preprocessing Resize224+ToTensor --normalization imagenet \
   --dataset-name Caltech101 --dataset-location ./data
```

### Evaluating a Particul-OD2 detector
There are two methods for evaluating the quality of an Out-of-distribution detector:
1) Study the evolution of the average confidence value provided by the model against a transform of the input with
increasing intensity.
2) Compare the confidence values provided by the model on In-distribution and Out-of-distribution data

#### Applying transform on the input
`tools/od2/dataset_confidence.py` is a tool which compares the quality of Particul-OD2 confidence measure to the
maximum softmax value. Given a dataset, a transform and a set of intensity values, it computes the Spearman rank
correlation scores between the intensity of the transform and the average value returned by both metrics. Currently,
the tool supports the following transform:
* rotation: intensity values are given in degrees
* gaussian_blur: a gaussian kernel of size 3 with intensity values corresponding to the sigma value
* gaussian_noise: a gaussian noise with intensity values corresponding to the noise ratio
* brightness: brightness multiplier (between 0 and 1, default: 1)
* contrast: contrast ratio (greater than 0, default: 1)
* saturation: saturation ratio (greater than 0, default: 1)
* vertical_shift: a vertical rotation of the image with intensity values corresponding to the number of pixels
* horizontal_shift: a horizontal rotation of the image with intensity values corresponding to the number of pixels
```bash
./dataset_confidence.py \
    --od2-classifier models/resnet_caltech101.pth \
    --od2-detector models/od2_module_calibrated.pth \
    -o models/od2_module_calibrated.pth --use-labels \
    --device cuda:0 \
    --preprocessing Resize224+ToTensor --normalization imagenet \
    --dataset-name Caltech101 --dataset-location ./data
    --transform gaussian_noise --values 0 0.1 0.2 0.3 0.5 0.7 1.0
    --log output_file.csv
    --device cuda:0 -v
```

### Comparing In-distribution to Out-Of-Distribution datasets
`tools/od2/evaluate.py` is a tool which is used to pairwise compare a reference in-distribution dataset to one or
several out-of-distribution datasets. For each pair (reference, OoD dataset), it returns:
* the Area Under the Receiver Operating Characteristic (AUROC) curve (true positive rate v. false positive rate)
* the Area Under the Precision Recall (AUPR) curve (precision v. recall)
* the FPR80 score (false positive rate when true positive rate is 80%)

```bash
./tools/od2/evaluate.py \
  --preprocessing Resize224+ToTensor --normalization imagenet \
  --ind-name Caltech101 --ind-location ./data \
  --ood-name CIFAR100 CUB200 --ood-location ./data \
  --device cuda:0 \
  --plot --verbose \
  classifier-with-detectors \
  --od2-classifier models/resnet_caltech101.pth \
  --od2-detector models/od2_module_calibrated.pth
  ```
Note: this tool can also be applied to fine-grained Particul detectors that are trained using the procedure described
[here](#learning-pattern-detectors).

```bash
./tools/od2/evaluate.py \
  --preprocessing Resize224+ToTensor --normalization imagenet \
  --ind-name CUB200 --ind-location ./data \
  --ood-name CIFAR100 Caltech101 --ood-location ./data \
  --device cuda:0 \
  --plot --verbose \
  detectors-only --model models/calibrated_model.pth
  ```
