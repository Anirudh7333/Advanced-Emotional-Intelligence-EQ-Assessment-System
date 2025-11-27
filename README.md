# EQ reflection sandbox

I put this together to explore how far a plain Django app plus HuggingFace transformers can go for EQ-style prompts. Nothing fancy, just a single-page flow that asks for context, gathers long-form answers, then runs sentiment/emotion scoring before spitting back a dashboard.

## Setup

```
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on mac/linux
pip install -r requirements.txt
python manage.py migrate
```

## Run

```
python manage.py runserver
```

Then hit `http://127.0.0.1:8000/` and walk through the prompts. First run downloads the transformer weights, so I usually give it a minute.

## Extras

- `python manage.py createsuperuser` if you want the Django admin.
- `python manage.py test` runs the small regression checks I keep around.
