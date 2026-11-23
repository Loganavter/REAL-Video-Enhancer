# A UI module that puts a request on the wire at module import: the page layer's contract is build widgets, format
# text, emit signals, and a network call makes the module untestable without the network.
# provenance: src/ui/HomeTab.py:1-3 @ 1edc7c8d
# provenance-note: the legacy tab module, cut before the page split (the rule's docstring explains why the legacy
#   tabs are the migration's subject). The download and the backend subprocess live in src/ui/dialogs on purpose;
#   pages and shell are the scope this rule protects.
#@rule: page-process-http
#@findings: 1

import os

import requests
