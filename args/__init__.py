import argparse


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg: String representing a bool.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_list(arg):
    """Convert an argument string into a list.

    Args:
        arg: String representing a list.

    Returns:
        List value for the string.
    """
    if arg is None:
        return []
    else:
        return [int(x) for x in arg.split(',')]


def generate_parser():
    parser = argparse.ArgumentParser(description="Argument parser for the experiment.")

    # --------------------------------
    # General experiment parameters
    parser.add_argument('--data', type=str, default="SEER_stomach",
                        choices=["VALCT", "DLBCL", "PBC", "GBM", "NACD", "gbsg", "METABRIC", "SUPPORT",
                                 "SEER_liver", "SEER_brain", "SEER_stomach",
                                 ],
                        help="Dataset name for the experiment.")
    parser.add_argument('--n_exp', type=int, default=10,
                        help="Number of experiments to run.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")
    parser.add_argument('--model', type=str, default="DeepHit",
                        choices=["MTLR", "DeepHit", "CoxPH", "AFT", "GB",
                                 "CoxTime", "CQRNN", "LogNormalNN", "KM"],
                        help="Model name.")
    # --------------------------------
    # Conformalize parameters
    parser.add_argument('--error_f', type=str, default="Quantile",
                        choices=["Quantile"],
                        help="Error function to use. Deprecated: 'Quantile' only.")
    parser.add_argument('--decensor_method', type=str, default="sampling",
                        choices=["uncensored", "margin", "PO", "sampling"],
                        help="Decensoring method to use when 'error_f' = Quantile.")
    parser.add_argument('--mono_method', type=str, default="bootstrap",
                        choices=["ceil", "floor", "bootstrap"],
                        help="Method to make the quantile prediction monotonic.")
    parser.add_argument('--interpolate', type=str, default="Pchip",
                        choices=["Linear", "Pchip"],
                        help="Interpolation method.")
    parser.add_argument('--n_quantiles', type=int, default=19,
                        help="Number of quantiles to use for the quantile prediction."
                             "reasonable numbers: 4, 9, 19, 39, 49, 99")
    parser.add_argument('--use_train', type=str_to_bool, default=True,
                        help="Whether to use the training set for calibration.")
    # --------------------------------
    # Network Structure parameters. Used for CoxPH, MTLR, and DeepHit only.
    parser.add_argument('--neurons', type=str_to_list, default=[64,64],
                        help="Hidden neurons in neural network. No space between numbers.")
    parser.add_argument('--norm', type=str_to_bool, default=True,
                        help="Whether to use batch norm in neural network.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout rate.")
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="Activation function. Possible values: Sigmoid, Tanh, ReLU, LeakyReLU, PReLU, ELU, SELU,"
                             "See https://pytorch.org/docs/stable/nn.html for more choices.")

    # --------------------------------
    # Training parameters, used for CoxPH, MTLR, and DeepHit only.
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help="Maximum number of training epochs. ")
    parser.add_argument('--early_stop', type=str_to_bool, default=True,
                        help="Whether to use early stop during training.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help="Term for weight decay (L2 penalty).")
    parser.add_argument('--lam', type=float, default=0,
                        help="Regularization scale for d-calibration. Only use for LogNormalNN.")

    # --------------------------------
    # Display parameters, used for CoxPH, MTLR, and DeepHit only.
    parser.add_argument('--verbose', type=str_to_bool, default=True,
                        help="Whether to print training information.")

    args = parser.parse_args()
    return args
