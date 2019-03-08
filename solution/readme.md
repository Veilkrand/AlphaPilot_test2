# Submissions - Velocity Vector Team

## Submission

1. Create a folder called `checkpoints`, and copy and *rename* your checkpoint into it as `checkpoint.pth`
2. Create a folder called `input_images`, and copy the images on which to generate the submission json.
3. Generate the json of predictions by running:

    ```bash
    python3 generate_submission.py
    ```

4. For other's use, copy the generated `submission.json` into folder `submission_json/` and rename appropriately.

## Scoring

1. First, generate submission file as outlined above.
2. Check score:

    ```bash
    python3 score_detections.py -g training_GT_labels_v2.json -p submission.json
    ```
