from cyborg.seedwork.domain.entities import BaseDomainEntity


class OAuthApplicationEntity(BaseDomainEntity):

    @classmethod
    def new_application(
            cls, name: str, client_id: str, client_secret: str
    ) -> 'OAuthApplicationEntity':
        return cls(raw_data={
            'name': name,
            'client_id': client_id,
            'client_secret': client_secret,
        })
