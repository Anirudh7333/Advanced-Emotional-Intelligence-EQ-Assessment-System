"""Simple WSGI entry point that serves the EQ assessment app."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eq_system.settings')

application = get_wsgi_application()


