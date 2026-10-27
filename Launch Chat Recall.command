#!/bin/zsh
cd -- "${0:A:h}"
exec python3 -m chat_recall serve --open
