"""ASGI hook I use for running the project with async servers."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eq_system.settings')

application = get_asgi_application()


