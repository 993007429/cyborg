import argparse
import atexit
import builtins
import os
import readline
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


shell_history_file = os.path.join(os.environ['HOME'], '.seal_shell.history')
readline.parse_and_bind('tab: complete')
try:
    readline.read_history_file(shell_history_file)
except IOError:
    pass  # It doesn't exist yet.


def shell_exit_callback(file_path=shell_history_file):
    try:
        import readline
        readline.write_history_file(file_path)
    except Exception as e:
        print('Unable to save Python command history: ', e)


atexit.register(shell_exit_callback)


def displayhook(value):
    if value is None:
        return
    builtins._ = None  # Set '_' to None to avoid recursion
    builtins._ = value


sys.displayhook = displayhook
del builtins

# Colorize the prompts if possible, Ubuntu tested
sys.ps1 = '\033[1;32m>>>\033[0m '
sys.ps2 = '\033[1;32m...\033[0m '  # for long line break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, default='')
    args = parser.parse_args()
    try:
        os.environ[args.environment] = 'true'
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        shell = TerminalInteractiveShell(user_ns=locals())
        shell.mainloop()
    except ImportError:
        print('WARNING: Loading InteractiveShell failed fall into default shell')
        import code
        shell = code.InteractiveConsole(locals=locals())
        shell.interact()
