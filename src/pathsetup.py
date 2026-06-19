"""
Add every src/ subfolder to sys.path so that flat import names (e.g.
`from movement_dataset import ...`) work regardless of which subfolder
the importing file lives in.

Import this module once at the top of any entry-point script:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import pathsetup  # noqa
"""
import os, sys

_src = os.path.dirname(os.path.abspath(__file__))

# Top-level src/ subdirs
for _sub in ('', 'extraction', 'latent_adapter', 'clf_finetuning', 'crosschannel_rf', 'shared', 'eval'):
    _d = os.path.join(_src, _sub)
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

# All v*/ experiment subdirs under extraction/ (architectures + datasets live there)
_extraction = os.path.join(_src, 'extraction')
if os.path.isdir(_extraction):
    for _entry in sorted(os.listdir(_extraction)):
        _d = os.path.join(_extraction, _entry)
        if os.path.isdir(_d) and not _entry.startswith('_') and _d not in sys.path:
            sys.path.insert(0, _d)
