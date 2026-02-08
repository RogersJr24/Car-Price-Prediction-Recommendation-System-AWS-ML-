import argparse
import os
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    # SageMaker passes hyperparameters as CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--colsample_bytree", type=float)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--eval_metric", type=str)
    parser.add_argument("--num_round", type=int)
    args = parser.parse_args()

    # SageMaker provides training data via this env var
    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    file_name = os.listdir(train_dir)[0]

    # IMPORTANT:
    # - No header
    # - Label is FIRST column
    df = pd.read_csv(os.path.join(train_dir, file_name), header=None)

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Use numpy arrays → avoids feature name issues
    dtrain = xgb.DMatrix(X.values, label=y.values)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "objective": args.objective,
        "eval_metric": args.eval_metric
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round
    )

    # SageMaker automatically uploads everything in this dir to S3
    model_dir = os.environ["SM_MODEL_DIR"]
    model.save_model(os.path.join(model_dir, "xgboost-model"))

if os.path.exists('train.py'):
    print("SUCCESS: train.py was written to disk.")
else:
    print("ERROR: train.py not found!")
