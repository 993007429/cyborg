from typing import Optional

from pydantic import BaseModel


class OpenAPIClient(BaseModel):

    app_name: str
    access_key: str
    secret_key: str
    enterprise_id: Optional[int] = None
