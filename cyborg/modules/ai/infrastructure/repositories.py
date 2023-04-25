from datetime import datetime
from typing import Optional, List

from sqlalchemy import desc

from cyborg.infra.session import transaction
from cyborg.modules.ai.domain.entities import AITaskEntity, AIStatisticsEntity, TCTProbEntity
from cyborg.modules.ai.domain.repositories import AIRepository
from cyborg.modules.ai.domain.value_objects import AITaskStatus
from cyborg.modules.ai.infrastructure.models import AIModel, TemplateModel, AITaskModel, AIStatisticsModel, TCTProbModel
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.seedwork.infrastructure.repositories import SQLAlchemyRepository


class SQLAlchemyAIRepository(AIRepository, SQLAlchemyRepository):

    @transaction
    def save_ai_task(self, ai_task: AITaskEntity) -> bool:
        model = self.convert_to_model(ai_task, AITaskModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        ai_task.update_data(**model.raw_data)
        return True

    def get_ai_task_by_id(self, task_id: int) -> Optional[AITaskEntity]:
        model = self.session.query(AITaskModel).get(task_id)
        return AITaskEntity.from_dict(model.raw_data) if model else None

    def get_latest_ai_task(self, case_id: int, file_id, ai_type: AIType) -> Optional[AITaskEntity]:
        model = self.session.query(AITaskModel).filter_by(
            case_id=case_id, file_id=file_id, ai_type=ai_type.value).order_by(desc(AITaskModel.id)).first()
        return AITaskEntity.from_dict(model.raw_data) if model else None

    def get_latest_calibrate_ai_task(self) -> Optional[AITaskEntity]:
        model = self.session.query(AITaskModel).filter_by(is_calibrate=True).order_by(desc(AITaskModel.id)).first()
        return AITaskEntity.from_dict(model.raw_data) if model else None

    def get_ai_task_ranking(self, task_id: int) -> Optional[int]:
        rows = self.session.query(AITaskModel.id).filter(
            AITaskModel.status == AITaskStatus.default,
            AITaskModel.expired_at > datetime.now()
        ).order_by(AITaskModel.id).all()
        for idx, row in enumerate(rows):
            if row[0] == task_id:
                return idx
        return None

    def get_ai_id_by_type(self, ai_type: AIType) -> Optional[int]:
        row = self.session.query(AIModel.id).filter_by(ai_name=ai_type.value).first()
        return row[0] if row else None

    def get_ai_name_by_template_id(self, template_id: int) -> Optional[str]:
        row = self.session.query(AIModel.ai_name).join(TemplateModel, AIModel.id == TemplateModel.ai_id).filter(
            TemplateModel.id == template_id).first()
        return row[0] if row else None

    def get_template_id_by_ai_name(self, ai_name: str) -> Optional[int]:
        row = self.session.query(TemplateModel.id).join(AIModel, TemplateModel.ai_id == AIModel.id).filter(
            AIModel.ai_name == ai_name).first()
        return row[0] if row else None

    @transaction
    def save_ai_stats(self, stats: AIStatisticsEntity) -> bool:
        model = self.convert_to_model(stats, AIStatisticsModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        stats.update_data(**model.raw_data)
        return True

    def get_ai_stats(self, ai_type: AIType, company: str, date: str) -> Optional[AIStatisticsEntity]:
        model = self.session.query(AIStatisticsModel).filter_by(
            ai_type=ai_type.value,
            company=company,
            date=date
        ).order_by(desc(AIStatisticsModel.id)).first()
        return AIStatisticsEntity.from_dict(model.raw_data) if model else None

    @transaction
    def save_tct_prob(self, prob: TCTProbEntity) -> bool:
        model = self.convert_to_model(prob, TCTProbModel)
        if not model:
            return False
        self.session.add(model)
        self.session.flush([model])
        prob.update_data(**model.raw_data)
        return True

    def get_tct_prob(self, slice_id: int) -> Optional[TCTProbEntity]:
        model = self.session.query(TCTProbModel).filter_by(
            slice_id=slice_id
        ).order_by(desc(TCTProbModel.id)).first()
        return TCTProbEntity.from_dict(model.raw_data) if model else None

    def get_tct_probs_by_slices(self, slices: List[dict]) -> List[TCTProbEntity]:
        mapping = {s['id']: s for s in slices}
        models = self.session.query(TCTProbModel).filter(TCTProbModel.slice_id.in_(mapping.keys())).all()
        return [TCTProbEntity.from_dict(
            model.raw_data, check_result=mapping.get(model.slice_id, '')) for model in models]
