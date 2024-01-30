from src.argparser import arg_parser


def test_arg_parser_no_args():
    parser = arg_parser()
    args = parser.parse_args([])
    assert args.cpus == 10
    assert args.command is None


def test_arg_parser_sim_command():
    parser = arg_parser()
    args = parser.parse_args(["sim"])
    assert args.command == "sim"
    assert args.new_data is False
    assert args.new_features is False
    assert args.chrs == "1"
    assert args.window_size == 50
    assert args.max_cnv_length == 100000
    assert args.min_cnv_gap == 50
    assert args.N_percentage == 0.7
    assert args.cov == 10


def test_arg_parser_train_command():
    parser = arg_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.EDA is False


def test_arg_parser_sim_command_with_args():
    parser = arg_parser()
    args = parser.parse_args(
        [
            "sim",
            "--new_data",
            "--new_features",
            "--chr",
            "2",
            "3",
            "4",
            "--window_size",
            "100",
            "--max_cnv_length",
            "200000",
            "--min_cnv_gap",
            "100",
            "--n_percentage",
            "0.8",
            "--cov",
            "20",
        ]
    )
    assert args.command == "sim"
    assert args.new_data is True
    assert args.new_features is True
    assert args.chrs == ["2", "3", "4"]
    assert args.window_size == 100
    assert args.max_cnv_length == 200000
    assert args.min_cnv_gap == 100
    assert args.N_percentage == 0.8
    assert args.cov == 20


def test_arg_parser_train_command_with_args():
    parser = arg_parser()
    args = parser.parse_args(["train", "--eda"])
    assert args.command == "train"
    assert args.EDA is True
