def update_clarity(slice_id: int, slice_file_path: str):
    from cyborg.app.service_factory import AppServiceFactory
    AppServiceFactory.slice_service.update_clarity(slice_id, slice_file_path)
