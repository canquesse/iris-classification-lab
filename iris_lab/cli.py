import argparse
import json
from .experiment import run_experiment, predict_measurements


def main():
    parser=argparse.ArgumentParser(description="Train, compare and inspect introductory Iris classifiers")
    commands=parser.add_subparsers(dest="command",required=True)
    train=commands.add_parser("train",help="Run a reproducible experiment")
    train.add_argument("--output",default="reports")
    train.add_argument("--seed",type=int,default=42)
    predict=commands.add_parser("predict",help="Predict from four measurements in cm")
    predict.add_argument("measurements",type=float,nargs=4,metavar="CM")
    predict.add_argument("--model",default="reports/model.joblib")
    args=parser.parse_args()
    try:
        if args.command=="train":
            result=run_experiment(args.output,args.seed)
            print(f"Selected {result['selected_model']}; held-out accuracy {result['test_accuracy']:.3f}")
            print(f"Reports saved to {args.output}")
        else:
            print(json.dumps(predict_measurements(args.model,args.measurements),indent=2))
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))

if __name__=="__main__":
    main()
