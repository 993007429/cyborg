celery -A cyborg.celery.app worker -l info -c1 -Q ai_task --max-tasks-per-child=1 --time-limit=1800
