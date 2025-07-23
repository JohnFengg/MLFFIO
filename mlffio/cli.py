import argparse
from mlffio.subcommands import lasp_process

def main():
    parser=argparse.ArgumentParser(prog="mlffio", description="MLFFIO CLI Toolkit")
    subparsers=parser.add_subparsers(dest="command")

    lte_parser=subparsers.add_parser("lasptrainevt",help='Run LASP training epoch v.s. time')
    lte_parser.set_defaults(func=lasp_process.run_evt)

    ltl_parser=subparsers.add_parser("lasptrainloss",help='Run LASP training loss v.s. epoch')
    ltl_parser.set_defaults(func=lasp_process.run_loss)

    ltp_parser=subparsers.add_parser("lasptrainplot",help='Run LASP training loss plot')
    ltp_parser.set_defaults(func=lasp_process.run_plot)

    mp_parser=subparsers.add_parser("mdplot",help='Run MD e,f,msd v.s. steps')
    mp_parser.set_defaults(func=lasp_process.run_md)

    args=parser.parse_args()
    if hasattr(args,'func'):
        args.func(args)
    else:
        parser.print_help()