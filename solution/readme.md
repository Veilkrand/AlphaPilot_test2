# Submissions - Velocity Vector Team

## Submission

Setup the correct image folder with ranking images.
1. Create a folder called `checkpoints` and copy your checkpoint into it.
2. Rename the checkpoint as `checkpoint.pth`.
3. IF required, copy images for testing into `input_images`.
4. Create the json of predictions by running:

    ```bash
    python3 generate_submission.py
    ```
5. For other's use, copy the generated `submission.json` into `submission_json/` and rename appropriately.

## Scoring

1. First, generate submission file as outlined above.
2. Check score:

    ```bash
    python3 score_detections.py -g training_GT_labels_v2.json -p submission.json
    ```
