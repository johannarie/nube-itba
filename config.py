#!/usr/bin/env python3
import os

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID') if os.environ.get('AWS_ACCESS_KEY_ID') else 'YOUR_AWS_ACCESS_KEY'
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY') if os.environ.get('AWS_SECRET_ACCESS_KEY') else 'YOUR_AWS_SECRET_ACCESS_KEY'
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN') if os.environ.get('AWS_SESSION_TOKEN') else 'YOUR_AWS_SESSION_TOKEN'
BUCKET_NAME = os.environ.get('BUCKET_NAME') if os.environ.get('BUCKET_NAME') else 'model-pd'
FOLDER = os.environ.get('FOLDER') if os.environ.get('FOLDER') else 'modelos'