from cyborg.seedwork.domain.value_objects import BaseValueObject


class CurrentOauthClient(BaseValueObject):
    name: str
    client_id: str
    client_secret: str
