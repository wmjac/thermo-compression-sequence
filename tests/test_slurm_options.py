"""Every command's public options are explicit in its SLURM script."""
from pathlib import Path
import re
from mixture_models.cli import build_parser
from mixture_models.models import get_model_module


def test_all_cli_options_are_spelled_out_in_slurm():
    parser = build_parser(get_model_module().training_protocol())
    commands = next(action for action in parser._actions if hasattr(action, "choices") and isinstance(action.choices, dict))
    root = Path(__file__).resolve().parents[1] / "slurm"
    for command, filename in [("train", "train.slurm"), ("canonicalize", "canonicalize.slurm"), ("evaluate", "evaluate.slurm")]:
        text = (root / filename).read_text()
        supplied = set(re.findall(r"--[a-z][a-z0-9-]*", text))
        expected = {option for action in commands.choices[command]._actions
                    for option in action.option_strings if option != "--help" and option.startswith("--")}
        assert expected <= supplied, (command, expected - supplied)
