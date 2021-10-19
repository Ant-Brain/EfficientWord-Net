"""
.. include:: ../README.md
"""

import os
RATE=16000
samples_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)),"sample_refs")

def _install(verbose=False):
    import sys
    import subprocess
    output = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--index-url', 'https://google-coral.github.io/py-repo/', 'tflite_runtime'],
        check=False,
        stdout=sys.stdout if verbose else subprocess.PIPE,
        stderr=sys.stderr)
    # if verbose:
    #     print(output)
    output.check_returncode()

def _check_install(verbose=False, upgrade=False, force=False, **kw):
    USER_MESSAGE = (
        "NOTE: The reason that this is even necessary is because tensorflow still "
        "hasn't released tflite_runtime on pypi and pypi freaks out if "
        "a url outside of pypi is included as a dependency. "
        "Once this upstream issue is resolved this message will go away.")
    if upgrade or force:
        # print('tflite is installed, but checking for a newer version.')
        _install(verbose=verbose, **kw)
    try:
        import tflite_runtime
    except ImportError as e:
        print(e, 'installing the right version for your system now...')
        if verbose:
            print(USER_MESSAGE)
        _install(verbose=verbose, **kw)
        if verbose:
            print('.'*50)
        print('All done! Carry on.')

_check_install()
