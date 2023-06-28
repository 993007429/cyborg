from typing import List

from cyborg.celery.app import celery_task


@celery_task
def update_clarity(slice_id: int, slice_file_path: str):
    from cyborg.app.service_factory import AppServiceFactory
    AppServiceFactory.slice_service.update_clarity(slice_id, slice_file_path)


@celery_task
def create_report(case_id: str, report_id: str, report_data: str, jwt: str):
    from cyborg.app.service_factory import AppServiceFactory
    res = AppServiceFactory.slice_service.create_report(
        case_id=case_id, report_id=report_id, report_data=report_data, jwt=jwt)
    return res


@celery_task
def export_slice_files(client_ip: str, case_ids: List[str], path: str, need_db_file: bool = False):
    from cyborg.app.service_factory import AppServiceFactory
    return AppServiceFactory.slice_service.export_slice_files(
        client_ip=client_ip, case_ids=case_ids, path=path, need_db_file=need_db_file)
