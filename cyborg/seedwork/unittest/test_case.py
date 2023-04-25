from pydantic import BaseModel


class TestCase(BaseModel):
    params: dict
    expect_results: dict

    def to_tuple(self):
        return self.params, self.expect_results
