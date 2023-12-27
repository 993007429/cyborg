class CompanyAIThresholdUpdatedEvent(object):
    event_name = 'company_ai_threshold_updated'

    def __init__(self, params: dict, search_key: dict):
        self.params = params
        self.search_key = search_key
