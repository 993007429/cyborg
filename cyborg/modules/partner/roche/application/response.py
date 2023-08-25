from cyborg.seedwork.application.responses import AppResponse


class RocheAppResponse(AppResponse):

    def dict(self, *_, **__):
        if self.err_code:
            return {
                'error_code': self.err_code,
                'error_message': self.message or '',
            }
        else:
            return self.data
