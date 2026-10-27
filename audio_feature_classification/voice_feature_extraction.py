import os
import re
import subprocess

import numpy as np
import pandas as pd


class OpenSMILE:
    def __init__(self, root_dir, config_path):
        self.opensmile_dir = os.path.expanduser(root_dir)
        self.opensmile_exe = os.path.join(
            self.opensmile_dir,
            "bin/linux_x64_standalone_static/SMILExtract",
        )
        self.opensmile_conf = config_path
        self.verify_dependencies()

    def verify_dependencies(self):
        if not os.path.isdir(self.opensmile_dir):
            raise Exception(f"Can't find openSMILE home {self.opensmile_dir}")
        if not os.path.isfile(self.opensmile_conf):
            raise Exception(f"Can't find openSMILE config {self.opensmile_conf}")
        if not os.path.isfile(self.opensmile_exe):
            raise Exception(f"Can't find SMILExtract executable {self.opensmile_exe}")

        try:
            output = subprocess.check_output(
                ["ffmpeg", "-version"], stderr=subprocess.STDOUT
            )
            m = re.search(rb"ffmpeg version (.*?) Copyright", output)
            if m:
                print("ffmpeg version", m.group(1).decode())
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise Exception("Can't find ffmpeg executable") from e

        try:
            output = subprocess.check_output(
                [self.opensmile_exe, "-h"], stderr=subprocess.STDOUT
            )
            m = re.search(rb"openSMILE version (.*)", output)
            if m:
                print("openSMILE version", m.group(1).decode())
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise Exception("Can't run SMILExtract executable") from e

    def process(self, wav_path):
        features_file = os.path.join(os.path.dirname(wav_path), "temp.csv")
        command = [
            self.opensmile_exe,
            "-I", wav_path,
            "-C", self.opensmile_conf,
            "--csvoutput", features_file,
        ]

        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)

            features = pd.read_csv(features_file, sep=";")
            features.drop("name", axis=1, inplace=True)
            return features.values[0]
        finally:
            if os.path.exists(features_file):
                os.remove(features_file)