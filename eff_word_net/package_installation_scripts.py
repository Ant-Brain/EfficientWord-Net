def _install_tflite(verbose=False):
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

def check_install_tflite(verbose=False, upgrade=False, force=False, **kw):
    """
    Installs tflite_runtime package since tflite_runtime cannot be installed by
    mentioning in requirements.txt
    """
    USER_MESSAGE = (
        "NOTE: The reason that this is even necessary is because tensorflow still "
        "hasn't released tflite_runtime on pypi and pypi freaks out if "
        "a url outside of pypi is included as a dependency. "
        "Once this upstream issue is resolved this message will go away.")
    if upgrade or force:
        # print('tflite is installed, but checking for a newer version.')
        _install_tflite(verbose=verbose, **kw)
    try:
        import tflite_runtime
    except ImportError as e:
        print(e, 'installing the right version for your system now...')
        if verbose:
            print(USER_MESSAGE)
        _install_tflite(verbose=verbose, **kw)
        if verbose:
            print('.'*50)
        print('All done! Carry on.')

def _install_librosa(verbose=False):
    import sys
    import subprocess
    output = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'librosa'],
        check=False,
        stdout=sys.stdout if verbose else subprocess.PIPE,
        stderr=sys.stderr)
    # if verbose:
    #     print(output)
    output.check_returncode()

def check_install_librosa(verbose=False, upgrade=False, force=False, **kw):
    """
    Installs librosa package on demand while generating references since,
    for some platforms librosa binaries arent available eg: Raspberry Pi 64 bit

    Librosa isnt required to perform inference over generated reference files,
    hence this will allow users to generate reference files in platforms where librosa binaries
    are available and use it in platforms where librosa binaries arent available
    """
    USER_MESSAGE = (
        "NOTE: The reason that this is even necessary is because tensorflow still "
        "hasn't released tflite_runtime on pypi and pypi freaks out if "
        "a url outside of pypi is included as a dependency. "
        "Once this upstream issue is resolved this message will go away.")
    if upgrade or force:
        # print('tflite is installed, but checking for a newer version.')
        _install_librosa(verbose=verbose, **kw)
    try:
        import librosa
    except ImportError as e:
        print(e, 'installing the right version for your system now...')
        if verbose:
            print(USER_MESSAGE)
        _install_librosa(verbose=verbose, **kw)
        if verbose:
            print('.'*50)
        print('All done! Carry on.')

