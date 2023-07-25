import sys
import time

from cyborg.app.request_context import request_context
from cyborg.app.service_factory import AppServiceFactory


if __name__ == '__main__':
    while True:
        with request_context:
            updated = AppServiceFactory.ai_service.maintain_ai_tasks().data
            if updated:
                print(f'处理超时任务：{updated}')
            else:
                print('.')
            sys.stdout.flush()
            time.sleep(60)
