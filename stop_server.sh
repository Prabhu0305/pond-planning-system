#!/bin/bash
pkill -f 'app.py' && echo "Pond Planning server stopped." || echo "No server running."
