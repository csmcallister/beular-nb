import argparse
import base64
from io import StringIO
import json
import os


import eli5
from eli5.formatters import format_as_html
from eli5.formatters.as_dict import format_as_dict
import joblib
import nltk
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker
)

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

from train import randomized_grid_search


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the env vars
    parser.add_argument(
        '--output-data-dir',
        type=str,
        default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ['SM_MODEL_DIR']
    )
    parser.add_argument(
        '--train',
        type=str,
        default=os.environ['SM_CHANNEL_TRAIN']
    )

    args = parser.parse_args()

    train_files = [
        os.path.join(args.train, f) for f in os.listdir(args.train)
        if f.endswith('.csv')
    ]

    if len(train_files) == 0:
        raise ValueError((f'There is no file in {args.train}.'))
    elif len(train_files) != 1:
        raise ValueError((f'There is more than one file in {args.train}'))

    train_file = train_files[0]

    # Train Data
    df = pd.read_csv(train_file)
    df.columns = ['Clause Text', 'Classification']
    df = df.astype({'Classification': np.float64, 'Clause Text': str})
    X = df['Clause Text']
    y = df['Classification']

    print("Fitting model...")
    model = randomized_grid_search(X, y, n_iter=1000)
    print("Done fitting model!")
    joblib.dump(
        model,
        os.path.join(args.model_dir, "model.joblib")
    )
    print("Done saving the model!")


def input_fn(input_data, content_type):
    """Parse input data payload
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == 2:
            # This is a labelled example, which includes the target
            df.columns = ['Clause Text', 'Classification']
            df = df.astype({'Classification': np.float64, 'Clause Text': str})
        elif len(df.columns) == 1:
            # This is an unlabelled example.
            df.columns = ['Clause Text']
            df = df.astype({'Clause Text': str})
        else:
            raise ValueError(
                "Invalid payload. Payload must contain either 3 columns \
                (id, text, target) or one column (text)"
            )

        return df

    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(inferences, accept):
    """Format inferences output
    The default accept/content-type b/w containers for inference is JSON.
    We also want the ContentType/mimetype the same value as accept so the
    next container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for inference in inferences.tolist():
            if len(inference) == 4:
                target, pred_prob, prediction, expl = inference
                instances.append({
                    "pred_prob": pred_prob,
                    "prediction": prediction,
                    "target": target,
                    "expl": expl
                })
            elif len(inference) == 3:
                pred_prob, prediction, expl = inference
                instances.append({
                    "pred_prob": pred_prob,
                    "prediction": prediction,
                    "expl": expl
                })
            else:
                pred_prob, prediction = inference
                instances.append({
                    "pred_prob": pred_prob,
                    "prediction": prediction
                })
        json_output = {"instances": instances}
        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(
            encoders.encode(prediction, accept), mimetype=accept
        )
    else:
        raise RuntimeError(f"{accept} is not supported by this script.")


def predict_fn(input_data, model):
    """Call predict on the estimator given input data.
    """
    input_data = input_data['Clause Text']

    try:
        lsa = model.named_steps['lsa']
        lsa_passthrough = lsa.get_params().get('passthrough', False)
    except KeyError:
        lsa_passthrough = False

    if lsa_passthrough:
        inferences = explain_pred(input_data, model)
    else:
        inferences = predict_from_model(input_data, model)

    if 'Classification' in input_data:
        # Return the label (as the first column) alongside the inferences
        return np.insert(inferences, 0, input_data['Classification'], axis=1)
    else:
        return inferences


def model_fn(model_dir):
    """Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def explain_pred(input_data, model):
    y_preds = []
    y_probs = []
    encoded_htmls = []
    for i in input_data:
        expl = eli5.explain_prediction(
            model.steps[-1][1],
            i,
            model.steps[0][1],
            target_names=['Compliant', 'Not Compliant'],
            top=10
        )
        html_explanation = format_as_html(
            expl,
            force_weights=False,
            show_feature_values=True
        ).replace("\n", "").strip()
        encoded_html = base64.b64encode(bytes(html_explanation, encoding='utf-8'))
        encoded_htmls.append(encoded_html)
        expl_dict = format_as_dict(expl)
        targets = expl_dict['targets'][0]
        target = targets['target']
        y_pred = 1 if target.startswith('N') else 0
        y_prob = targets['proba']
        if len(i.split()) < 3:
            # one or two words can't be non-compliant
            y_pred = 0
            y_prob = 1.0
        y_prob = f'{round(y_prob, 3) * 100}%'
    y_preds.append(y_pred)
    y_probs.append(y_prob)
    inferences = np.column_stack((y_probs, y_preds, encoded_htmls))

    return inferences


def predict_from_model(input_data, model):
    y_preds = model.predict(input_data)

    positive_class_idx = list(model.classes_).index(1)
    try:
        y_scores = model.predict_proba(input_data)[:, positive_class_idx]
    except AttributeError:
        y_scores = model.decision_function(input_data)
    inferences = np.column_stack((y_scores, y_preds))

    return inferences
