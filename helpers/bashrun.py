### Execute a bash command and return the output
import subprocess


def bashrun(command):
    # execute command
    process = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Get the command output
    return process.stdout