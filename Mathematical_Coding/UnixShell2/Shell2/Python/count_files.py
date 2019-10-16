#!/home/mark/anaconda3/bin/python
import subprocess

subprocess.call("find .. -type f | wc -l", shell=True)
