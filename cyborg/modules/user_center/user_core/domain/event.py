class CompanyAIThresholdUpdatedEvent(object):

    event_name = 'company_ai_threshold_updated'

    def __init__(self, threshold_range: int, threshold_value: float):
        self.threshold_range = threshold_range
        self.threshold_value = threshold_value
