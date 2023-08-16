## Fields vs Roads classification, detection, and segmentation


## Demo on a test video

<div align="left">
      <a href="[https://www.youtube.com/watch?v=5yLzZikS15k](https://www.youtube.com/watch?v=7ruIdzj4COc)">
         
      </a>
</div>



## Installation 

Create your virtual environment using Python 3.9 then install this project's dependencies using:

`pip install -r requirements.txt`

`wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`







## Classification

**Looking at the data**

During the inspection of the dataset, it has been noted that some images belonging the class "roads" are 
present in the "fields" folder, we will start by cleaning the dataset to have accurate representations.

![Screenshot of the noisy dataset.png](doc%2FScreenshot%20of%20the%20noisy%20dataset.png)

The size of the dataset is abysmal, we will need to perform data augmentation, but let's start with a very basic 
approach the have an idea of the complexity of the task, because although this task appear mondane as it is very easy to 
discriminate between roads and fields, the quality of the dataset may pose a problem


![performance of the first training.png](doc%2Fperformance%20of%20the%20first%20training.png)

As expected the model has the over-fitting issues due to the size of the data, let's try some very common approaches to 
mitigate this issue:
- Dropout; this will exclude some neurons from activating, introducing a noise in the network, the model will generalize
better when he is able to predict good results with lower quantities of characteristics.
- Regularization; this will hinder the model's ability to learn, reducing the speed with which it will start to over fit.
- Variable Learning-rate; the learning rate will vary from the initial learning rate (lr0) to the final learning rate (lrf)
this will allow the model to better find the global minimums during the gradiant decent in the back-propagation phase.



The performance of the second model (in blue)

**Training code**

`python basic_classification.py --data_dir dataset/ --num_classes 2 --batch_size 16 --num_epochs 37 --lr0 0.001 --lrf 0.00001 --gpu`

![perfomrance of the second training.png](doc%2Fperfomrance%20of%20the%20second%20training.png)

The second model appears to be the best we can do given this data, and even if the results look perfect, we should still 
introduce data augmentation even if it will reduce our accuracy on the testing set, it will still increase our model's
overall generalization capabilites.

![performance_augmented_model.png](doc%2Fperformance_augmented_model.png)
**inference code**

`python basic_classification_inference.py --model_path models/lr0_0.001_lrf_1e-05_batch_16_exp_1/lr0_0.001_lrf_1e-05_batch_16_best_model.pth --test_image_dir dataset/test_images/ --gpu`

![test_images_output.png](doc%2Ftest_images_output.png)

The script also outputs a CSV with the detected classes from the `dataset/test_images/` folder 
and put here `inference_results.csv`

And you will also creat a image with up to 10 random images here `random_images.png`

The classification model is available here `models/lr0_0.001_lrf_1e-05_batch_16_exp_1/lr0_0.001_lrf_1e-05_batch_16_best_model.pth`

The results are decent, yet, we discover a severe problem in this project; *What if we have both fields AND roads 
in an image ?*
as in image 6.jpeg

This is leading to the bulk of our project: **Detection** and **Segmentation**

## Detection

Transformers are good, but heavy. While YOLOs are fast and not so much worse, leading to this:

workflow:
-
- Use Stable Diffusion 2 to augment our dataset with synthetic images with a road and a field both present (highly experimental).
- Use ViTs to predict on images, then SAM (Segment Anything from Meta) to creat a Dataset.
- Train a Yolov7 to detect roads and fields .
- Optimise the model with latest State Of The Art techniques.
- Evaluate everything.



Step one: POC; let's check that we can reliably transform classification images to Detection/Segmentation using owl-vin 
and SAM

![6.jpeg](doc%2F6.jpeg)
![owlvit_box-6.jpg](doc%2Fowlvit_box-6.jpg)
![owlvit_segment_anything_output-6.jpg](doc%2Fowlvit_segment_anything_output-6.jpg)

And again:

![1.jpeg](dataset%2Ftest_images%2F1.jpeg)
![owlvit_box-1.jpg](doc%2Fowlvit_box-1.jpg)
![owlvit_segment_anything_output-1.jpg](doc%2Fowlvit_segment_anything_output-1.jpg)

Astonishing work from these researchers !


Test it yourself !

`python demo.py --image_path dataset/test_images/6.jpeg --text_prompt field,road -o outputs/ --device cuda:0 --get_topk --box_threshold 0.25`

Let's go further beyond! Let's try it on a syntheticaly generated image for our use-case!
**One very big adventage for using synthetic data is to avoid RGPD problems**
*Note: The size of the project is getting out of control; I will not include the repo and the code for using Stable 
Diffusion to generate the synthetic code*

- Repo used: https://github.com/Stability-AI/stablediffusion
- Prompt used `python scripts/txt2img.py --prompt "ralistic image of a road passing by a harvesting hay field" --ckpt /home/aymen/Downloads/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 384 --W 384  --device cuda`
- output given:![stable_output.png](doc%2Fstable_output.png); some images can be used.

I really liked this generated image, so let's try it out:

![nice_synth.png](doc%2Fnice_synth.png)
![owlvit_box-39.jpg](doc%2Fowlvit_box-39.jpg)
![owlvit_segment_anything_output-38.jpg](doc%2Fowlvit_segment_anything_output-38.jpg)

It seems like CLIP does get it correctly but SAM doesn't, but to fair, not only this images doesn't look coherent to 
us, but even more so to the model
unless more time is invested in the fine tuning of the generation of the synthetic data, we are only going to use 
them as validation/test.



The repo owner also did an amazing combining OWL-ViT-CLIP with SAM and I only needed to adjust the code to handle 

confidence threshold
correctly.
Then I spent some time to adjust the script so it will also write the Detection output in Darknet format (coco) 
and also created a bash script that will
transform the given dataset into the detection dataset.

One sever drawback is that this method will introduce its error to the new model as it's output is treated a 
groundtruth and that OWL vit have 
troubles detecting on some images, we lost about 30% of our dataset training images due to this.


YOLOv7 Prediction using OWL-VIT annotated dataset:
  - 
First results:
- Decent model performance with the mAP@0.5 of 0.723 but with low accuracy
![First detection results.png](doc%2FFirst%20detection%20results.png)
- Test example:
  - CLIP annoitations:
![clip annotation.png](doc%2Fclip%20annotation.png)
  - YOLOv7 Predictions:
![Yolov7 detections.png](doc%2FYolov7%20detections.png)

The results aren't good enough, but is it an approach problem or data problem ?, let's annotate the data ourself and 
compare, of course, one
major problem when it comes on detection and segmentation on this project is: what is a field ? and what isn't ?, 
and, the same thing is applied to roads,
are dirt roads roads ? and are the grass fields fields ? or should we only consider harvesting fields ?.
In order to maximize the challenge, we will want to detect everything, no matter how hard to caracterize the object is, 
as we can always 
increase the confidence threshold to get the definitions that we want, but I want to be able to detect everything.
 
The dataset generated with CLIP and OWL-VIT will be named **detection_dataset** while the manually annotated one will 
be name **manual_detection**

I have trained the model using this repo:https://github.com/WongKinYiu/yolov7

One your environment ready, you can run the following command for training

`python train.py --weights yolov7.pt --data manual_detection/data.yaml --batch-size 4 --epochs 100 --hyp data/hyp.scratch.p5.yaml`

With lr0 and lrf edited from 0.01/0.1 to 0.001/0.05

And for testing

`python test.py  --weights runs/train/exp2/weights/best.pt --data /manual_detection/data.yaml`

I have uploaded the two dataset for visualisation here;

- https://app.roboflow.com/aymen-bouguerra-xcnsf/fields-roads-clip/1

- https://app.roboflow.com/aymen-bouguerra-xcnsf/field-road-manual/1


YOLOv7 Prediction using manually annotated dataset:
  - 
- Results:
- ![Results manual annotation.png](doc%2FResults%20manual%20annotation.png)
  - Annotation
![detection manual annotation .png](doc%2Fdetection%20manual%20annotation%20.png)
  - Predections
![Manual annotation prediction for detection.png](doc%2FManual%20annotation%20prediction%20for%20detection.png)

As we can see, the quantitative results are decent, but we can see that I was too ambitious in my annotations, 
but we can see in the qualitative results that the results are good, only one missed object.

We also notice that the model have lower overall performance when dealing with road, this is typically a bounding 
box problem; detection models are well known to be bad at dealing with objects with abstract shapes, it is difficult 
for the model to know understand the characteristics of the object inside the annotated bounding box if a large portion 
of that bounding box contains other objects, in my current company, we had have encountered this issue when wanting to 
detect two-wheelers such as bicycles and electric scooters, we found a bias where the model confidently and reliably 
detects this class only when a person is riding them, this is due the presence of persons in the annotated data as well 
as persons being more compact that bicycles.


The best solution (and a little bit more expensive) to deal with such objects is **segmentation**

## Segmentation

While playing with recent annotation models, I have come across a very easy-to-use tool that uses a 
recent project "DINO" I created a small script `dino_seg.py` that uses a function to *automatically* annotate 
images with the selected labels for segmentation and then saves the results in yolo format, I have loaded that dataset 
into the visualisation tool roboflow and have corrected some annotations and removes duplicates, the resulting dataset 
can be visualized here.

https://app.roboflow.com/aymen-bouguerra-xcnsf/inctance-segmentation-of-fields-and-raods-using-dino-as-annotator/7

You can download the dataset then run the script `create_yolo_lists_of_paths.py` to creat train.txt and val.txt and put 
them in the dataset folder, then edit the data.yaml paths to point to them (yolo format)

The masks that are composed of sharp and straight lines are made by me, while masks composed of hundreds, if not 
thousands of dots were made using DINO.


I then trained a YOLOv8 model (as its better for segmentation) on this dataset and the results are as follows:

*Note*: for training the YOLOv8 best practice is to clone the repo, install the environment and the dependencies then
use their scripts to train/test/detect ect., or you can simply run the `yolov8.py` script (more like function) provided
to train as a Proof Of Concept

*Note*: yolov8 also does detection at the same time of segmentation

For reference, here are the official Yolov8 results on COCO2017
![yolov8results.png](doc%2Fyolov8results.png)

- For yolov8-Large
  - Quantitative:
![yolov8l_quantitative.png](doc%2Fyolov8l_quantitative.png)

  - Qualitative:
     - Validation Annotations
![val_batch0_labels_yolov8l.jpg](doc%2Fval_batch0_labels_yolov8l.jpg)
     - Predictions
![val_batch0_pred_yolov8l.jpg](doc%2Fval_batch0_pred_yolov8l.jpg)


- For yolov8-Medium
  - Quantitative:
![yolov8Mediumresults.png](doc%2Fyolov8Mediumresults.png)

  - Qualitative:
     - Predictions
![yolov8preds.png](doc%2Fyolov8preds.png)


- For yolov8-Small
  - Quantitative:
![yolov8smallresults.png](doc%2Fyolov8smallresults.png)

  - Qualitative:
     - Predictions
![yolov8smallpreds.png](doc%2Fyolov8smallpreds.png)


- For yolov8-Tiny
  - Quantitative:
![yolov8tinyresults.png](doc%2Fyolov8tinyresults.png)

  - Qualitative:
     - Predictions
![val_batch0_pred_yolov8l.jpg](doc%2Fval_batch0_pred_yolov8l.jpg)

## ROC / f1 / P / R curves and all other kinds of metrics for each yolov7/yolov8 models are in runs/

## Inference Optimisation

As expected the models with a higher number of parameters tend to perform better, learning to a difficult compromise 
between speed and accuracy, leading to the subject of my current internship: Real time detection optimisation, where 
the goal is to keep using the heavier models and optimise them using state of the art compression techniques 
without degrading their accuracy significantly.

Here is a slide from one of my presentations where I had to vulgarise the optimisation approaches that I was trying to 
apply to their modes
(My internship was 100% done in French at La Défence)
![slide_internship.png](doc%2Fslide_internship.png)

*Note*: The comparisons are made to their already heavily optimised models using TensorRT FP16 and are only valid
on NVIDIA GPUs. I introduced during my internship Int8 QAT quantization and channel pruning, copressing the model by up 
to 75% and using INT8 datatypes.



## FAQ / Justifications

- Why did I choose RensNet50 as the classification model in the classification task ?
  - ResNet50s are very communally used as backbones in other models, and since the classification task looked 
  simple enough, a light-weight (by modern standards) model like RensNet50 looked attractive, no to mention the great
  pakages and support it has. We could have also gone with VGGs / or even super lightweight models with very few 
  hidden layers but I cas more focused on the detection/segmentation task.

- Why did I choose to use SGD optimizer with cosine learning rate momentum rather that another more common optimizer like ADAM ?
  - Even if ADAM remains a very well-rounded optimizer, I have used it since I read this 
  paper http://proceedings.mlr.press/v28/sutskever13.pdf that has 5k citations, this paper demonstrates how a well
  initialized network that uses a specific learning rate schedule outperforms other optimizers. 

- How did I choose the initial and final learning rate ? (schedule)
  - Mostly empirically; I start by only focusing on the initial learning rate, I try different values and stop when I
  find a values that makes the model converge on the few starting epochs. Then I fix that initial learning rate and 
  focus on the having decreasing cosine learning rate that **matches** the convergence speed of the model; we want to
  decrease the learning rate after each succesfull converging epoch so that the model can find new global minimums 
  without giving him a chance to "unlearn" by going back on is past steps.

- Why dod I choose YOLOs and not transformers ?
  - Transformers are extremely heavy models that are better, but that can take several second do perform a single
  inference, although they there are more and more technics such as int 4 quantization https://xiuyuli.com/qdiffusion/
  they are still a long road from being deployed in an embedded system while YOLOs are already heavily used in 
  production all over the world.

- The mdoels are detecting off road grass  as fields, is that intended ?
  - Yes, in order to add more challenge to this project, I wanted the resulting model to be able to also identify 
  not only havesting fields but also animal fields such as this one: ![img.png](doc%2Fimg.png) so I steared 
  the data nnotation with that mindset, and I also tought that we can always increase the confidence threshold
  to eliminate this cases if we needed to.




Please feel free to contact me if you have any other questions.

Also, feel free to sent me your classification testing data son I can convert it to segmentation testing data, there 
however, would be a problem of human bias. (What you consider to be a road or a field can differ for me)

I couldn't go into the details on how I trained the classification model and what I have done with stable diffusion 
and many other
core parts of this project as it would be too long to read. I would happily discuss iy with you if needed.
  
Thank you for reading.

## The paper review task is on Paper Reveiw.md in the project root





## Reference
Please give applause for [IDEA-Research](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/segment_anything) and [OWL-ViT on HuggingFace](https://huggingface.co/spaces/adirik/OWL-ViT)
and Open-vocabulary-Segment-Anything https://github.com/ngthanhtin/owlvit_segment_anything

## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@misc{minderer2022simple,
      title={Simple Open-Vocabulary Object Detection with Vision Transformers}, 
      author={Matthias Minderer and Alexey Gritsenko and Austin Stone and Maxim Neumann and Dirk Weissenborn and Alexey Dosovitskiy and Aravindh Mahendran and Anurag Arnab and Mostafa Dehghani and Zhuoran Shen and Xiao Wang and Xiaohua Zhai and Thomas Kipf and Neil Houlsby},
      year={2022},
      eprint={2205.06230},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
