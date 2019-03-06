# Datasets for AlphaPilot Competition

Details on test2 and all the available resources are available on AlphaPilot's [Test 2 page](https://www.herox.com/alphapilot/77-test-2).

### Final testing

- Each team’s folder of source code will be unzipped onto an instance of Ubuntu 16.04.5 LTS running in AWS type p3.2xlarge. The following instance will be used at the basis for the testing environment: https://aws.amazon.com/marketplace/pp/B077GCZ4GR  
- Team dependencies will be installed according to the libraries listed in a requirements.txt file. See ‘Algorithm Requirements’ section for more details on this file.
- The testing script, generate_submission.py, will run through all available image data, implement a function call to get the labels and execution time of team algorithms for a given image, and store the results for each image in a JSON file.
- Note: there is a cut-off for execution time for a team’s algorithm, and it is 10 seconds. If a team’s algorithm takes longer than this to output a label, the script will be stopped, and the team will be given an Algorithm Score of 0 points.
- Once all results are stored in a JSON, another testing script, score_detections.py, will compare the JSON file to a master JSON containing the ground-truth labels and will calculate the total algorithm score.
  - Note: there is a cut-off for the total submission evaluation time (Final Testing steps 3 and 4) and that is 2 hours. If a team’s algorithm takes longer than this to be evaluated, the testing will be stopped, and the team will be given an Algorithm Score of 0 points.

Separately, judges will read and score each team’s technical report which will be subsequently used to calculate their total final score for Test 2.

## [Training data - Images](https://www.herox.com/alphapilot/resource/314)

- Filename: Data_Training.zip

The Training Dataset contains roughly 9,300 images totaling 2.8GB. This will be the primary resource available to teams for development.


## Training data - Labels

### [Official Labels](https://www.herox.com/alphapilot/resource/317)

- Filename: training_GT_labels_v2.json

Labels provided by competition organizers. A JSON file containing the ground truth labels for the training dataset. Only the inner box (flyable area) is annotated.

### Our Labels

- Filename: export-2019-02-28T10_42_47.657Z.json

Manually annotated (mostly by Ayshine!) outer polygons (outer edge of the gate). (5000 - 68) = 4392 images. 68 were removed by labelbox due to free version restrictions.

## [Testing data - Images](https://www.herox.com/alphapilot/resource/318)

- filename: Data_LeaderboardTesting.zip

Test 2 - Leaderboard Testing images. This dataset contains roughly 1,000 images totaling 360MB. This test dataset should be used to see how well a team’s algorithm performs on unseen images. For the Leaderboard Testing Dataset, the ground truth is not provided; it is the team’s job to predict these labels.  
With this dataset, teams may submit a JSON file of their labels for scoring and placement on a leaderboard.

## [Starter Scripts](https://www.herox.com/alphapilot/resource/308)

- filename: starter_scripts_v2.zip

The starter scripts folder (starter_scripts_v2.zip) contains code to help teams create submissions:  

- `generate_results.py` – Sample submission with example class by which to define a solution. Also reads and plots images and labels.
- `generate_submission.py` – Test script that reads all test images, imports and calls a team’s algorithm, and outputs labels for all images in a JSON file
- `random_submission.json` – Sample JSON file of labels; this JSON is also the output of generate_submission.py and meets the submission requirements

## [Scorer Scripts](https://www.herox.com/alphapilot/resource/307)

- filename: scorer_scripts_v2.zip

Scorer scripts for submission evaluation. The scorer scripts folder (scorer_scripts_v2.zip) contains code to calculate a team’s Algorithm Score:  

- `score_detections.py` – Test script which calls a team’s JSON file, evaluates it against the ground-truth JSON file, and outputs a MAP score

To run the sample submission, configure the environment to match the environment described in the ‘Testing’ section. Teams will also need to install the following libraries which can be installed using pip:

- Shapely
- NumPy
Please note that these libraries should be compatible with Python 3.5.2.

&nbsp;  
&nbsp;  
The starter scripts and scorer scripts provided for Test 2 represent how each team’s source code will be tested. We will not be providing the exact test instance for teams. However if a team defines a GenerateFinalDetections() class with a predict(self,img) function that runs smoothly in generate_submission.py (as shown in the starter scripts), this provides the sanity checks needed on source code.  
  
The ground-truth (GT) labels represent the 4 coordinates which define a polygon that maximizes the flyable region of the AIRR gates. The primary measure used in the MAP score is the Intersection over Union (IoU). The IoU is computed by dividing the area of overlap between the GT and predicted bounding boxes by the area of union. For more information on the implementation of this metric used for AlphaPilot, read more here: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)

The leaderboard submissions should be passed in the format: 

```json
{"IMG XYZ": [[100,200,300,400]]}
```

### Additional Libraries Requirements

Algorithms must be compatible with Python 3.5.2. Teams may leverage open-source algorithms and data libraries to assist in their design. Use of TensorFlow is recommended but not required. No other specific software and hardware is required, but teams should review the testing information below for further considerations. Further, teams can choose any approach for their algorithms (i.e. teams don’t have to use machine learning).

Teams will need to list dependencies in a requirements.txt file included in their submitted archive which can be automatically generated. The requirements.txt file only needs to list libraries that need to be installed for an algorithm’s source code to run. This will be installed using pip with the following command:

```bash
>> pip install -r requirements.txt
```

Given this, all libraries listed in requirements.txt must be installed via pip. If teams want to use libraries that do not fit within this constraint, they can add their compiled versions of their libraries into the submitted archive. However, proceed with the latter option at risk, because the AlphaPilot team can only guarantee successful install of libraries using pip.

Note: GPU and CUDA drivers will already be installed in the testing environment. The following instance will be used at the basis for the testing environment: https://aws.amazon.com/marketplace/pp/B077GCZ4GR

### How is execution time of algorithms calculated?

The measure of execution time is assessed according to the wall clock time for the gate detector to read an image, extract the corners of the flyable region of the gate, and output the label. So, 2 seconds per image. Please also note the additional time restrictions during testing (see ‘Testing Section’ of Eye Exam).

- The measure of execution time is assessed according to the wall clock time for the `generate_submission.py` function to run `finalDetector.predict(img)` as defined by your class. That is, the time for the gate detector to read an image, extract the corners of the flyable region of the gate and output the label.
- The total Algorithm Score is then calculated by subtracting the average wall clock time from teh weighted MAP score and multipying by 35 to get a maximum of 70 points: `Score = 35 * (2 * MAP - avg_time)`


## Notes from Forum

1. We might have to remove 6 images, saw this post on forum:
```
I found these 6 image files in the random_submission.json {'IMG_0898.JPG', 'IMG_0980.JPG', 'IMG_1525.JPG', 'IMG_7276 (1).JPG', 'IMG_7277 (1).JPG', 'IMG_8548.JPG'} while the test data does not contain these 6 images.
In my leaderboard submission, there are total 1161 keys, which is slightly different from random_submission.json (1167). I'm wondering if this is a bug that caused my 0% leaderboard score (my flyable region detector definitely works).
```

2. This is the word from organizer:
```
Please check the random_submission.json in the starter scripts for an example and make sure you are following the same format for each image: "IMG_####.JPG": [[x1, y1, x2, y2, x3, y3, x4, y4, CS]]

Here is an example: 
"IMG_3668.JPG": [[370.3913304338685, 377.2051599830667, 7.742989976402814, 13.058747191455566, 246.50017198915896, 321.3924492511484, 342.11494534552276, 35.65517009139904, 0.5]]

Empty gates should have the format: [[]]
```