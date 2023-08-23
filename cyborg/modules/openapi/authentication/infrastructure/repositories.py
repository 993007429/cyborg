from typing import Type, Optional

from cyborg.modules.openapi.authentication.domain.entities import OpenAPIClientEntity
from cyborg.modules.openapi.authentication.domain.repositories import OpenAPIClientRepository
from cyborg.modules.openapi.authentication.infrastructure.models import OpenAPIClient


TRUSTED_OPENAPI_CLIENTS = {
    'roche': OpenAPIClient(**{
        'app_name': 'roche',
        'access_key': 'LRz0gQ5WYuRaHVDVYKSs2d2we9hBtXm2x4IEgBro',
        'secret_key': 'Sdhsnqow-eQNmQcCwMxxGTnGUdS4UHuBoMQB_10Qo6_jh3097p',
    }),
    'logene': OpenAPIClient(**{
        'app_name': 'logene',
        'access_key': '57iOffyuv39lb3OWk5osziYyP7PwMkaevZRIEQwF',
        'secret_key': 'DYp1jckdUGNtjIHB0LXvw15xkIdRa9lSvNfZstBfq0uGZI6A60',
    })
}


class ConfigurableOpenAPIClientRepository(OpenAPIClientRepository):

    @property
    def model_class(self) -> Type[OpenAPIClient]:
        return OpenAPIClient

    def get_by_app_name(self, app_name: str) -> Optional[OpenAPIClientEntity]:
        if not app_name:
            return None
        model = TRUSTED_OPENAPI_CLIENTS.get(app_name)
        return OpenAPIClientEntity.from_dict(model.dict()) if model else None
