from dependency_injector import containers, providers

from cyborg.app.request_context import request_context
from cyborg.modules.ai.application.services import AIService
from cyborg.modules.ai.domain.services import AIDomainService
from cyborg.modules.ai.infrastructure.repositories import SQLAlchemyAIRepository
from cyborg.modules.oauth.application.services import OAuthService
from cyborg.modules.oauth.domain.services import OAuthDomainService
from cyborg.modules.oauth.infrastructure.repositories import SqlAlchemyOAuthApplicationRepository
from cyborg.modules.openapi.authentication.application.services import OpenAPIAuthService
from cyborg.modules.openapi.authentication.domain.services import OpenAPIAuthDomainService
from cyborg.modules.openapi.authentication.infrastructure.repositories import ConfigurableOpenAPIClientRepository
from cyborg.modules.partner.roche.application.services import RocheService
from cyborg.modules.partner.roche.domain.services import RocheDomainService
from cyborg.modules.partner.roche.infrastructure.repositories import SQLAlchemyRocheRepository

from cyborg.modules.slice.application.services import SliceService
from cyborg.modules.slice.domain.services import SliceDomainService
from cyborg.modules.slice.infrastructure.repositories import SQLAlchemyCaseRecordRepository, \
    SQLAlchemyReportConfigRepository
from cyborg.modules.slice_analysis.application.services import SliceAnalysisService
from cyborg.modules.slice_analysis.domain.services import SliceAnalysisDomainService
from cyborg.modules.slice_analysis.infrastructure.repositories import SQLAlchemySliceMarkRepository, \
    SQLAlchemyAIConfigRepository
from cyborg.modules.user_center.user_core.application.services import UserCoreService
from cyborg.modules.user_center.user_core.domain.services import UserCoreDomainService
from cyborg.modules.user_center.user_core.infrastructure.repositories import SQLAlchemyUserRepository, \
    SQLAlchemyCompanyRepository


def create_request_context():
    return request_context


class Core(containers.DeclarativeContainer):

    request_context = providers.Factory(
        create_request_context
    )


class UserCenterContainer(containers.DeclarativeContainer):

    core = providers.DependenciesContainer()

    account_repository = providers.Factory(
        SQLAlchemyUserRepository, session=core.request_context.provided.db_session
    )

    company_repository = providers.Factory(
        SQLAlchemyCompanyRepository, session=core.request_context.provided.db_session
    )

    user_domain_service = providers.Factory(
        UserCoreDomainService,
        repository=account_repository,
        company_repository=company_repository
    )

    user_service = providers.Factory(
        UserCoreService,
        domain_service=user_domain_service,
    )


class SliceContainer(containers.DeclarativeContainer):

    core = providers.DependenciesContainer()

    user_center = providers.DependenciesContainer()

    case_record_repository = providers.Factory(
        SQLAlchemyCaseRecordRepository, session=core.request_context.provided.db_session
    )

    report_config_repository = providers.Factory(
        SQLAlchemyReportConfigRepository, session=core.request_context.provided.db_session
    )

    slice_domain_service = providers.Factory(
        SliceDomainService,
        repository=case_record_repository,
        report_config_repository=report_config_repository
    )

    slice_service = providers.Factory(
        SliceService,
        domain_service=slice_domain_service,
        user_service=user_center.user_service
    )


class SliceAnalysisContainer(containers.DeclarativeContainer):

    core = providers.DependenciesContainer()

    slice = providers.DependenciesContainer()

    _manual_slice_mark_repository = providers.Factory(
        SQLAlchemySliceMarkRepository,
        session=core.request_context.provided.slice_db_session,
        template_session=core.request_context.provided.template_db_session,
    )

    slice_mark_repository = providers.Factory(
        SQLAlchemySliceMarkRepository,
        session=core.request_context.provided.slice_db_session,
        template_session=core.request_context.provided.template_db_session,
        manual=_manual_slice_mark_repository
    )

    ai_config_repository = providers.Factory(
        SQLAlchemyAIConfigRepository,
        session=core.request_context.provided.db_session,
    )

    slice_analysis_domain_service = providers.Factory(
        SliceAnalysisDomainService,
        repository=slice_mark_repository,
        config_repository=ai_config_repository
    )

    slice_analysis_service = providers.Factory(
        SliceAnalysisService,
        domain_service=slice_analysis_domain_service,
        slice_service=slice.slice_service
    )


class AIContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    user_center = providers.DependenciesContainer()

    slice = providers.DependenciesContainer()

    slice_analysis = providers.DependenciesContainer()

    ai_repository = providers.Factory(
        SQLAlchemyAIRepository,
        session=core.request_context.provided.db_session
    )

    ai_domain_service = providers.Factory(
        AIDomainService,
        repository=ai_repository
    )

    ai_service = providers.Factory(
        AIService,
        domain_service=ai_domain_service,
        user_service=user_center.user_service,
        slice_service=slice.slice_service,
        analysis_service=slice_analysis.slice_analysis_service
    )


class OpenAPIContainer(containers.DeclarativeContainer):
    core = providers.DependenciesContainer()

    user_center = providers.DependenciesContainer()

    client_repository = providers.Factory(
        ConfigurableOpenAPIClientRepository
    )

    openapi_auth_domain_service = providers.Factory(
        OpenAPIAuthDomainService,
        client_repository=client_repository,
    )

    openapi_auth_service = providers.Factory(
        OpenAPIAuthService,
        domain_service=openapi_auth_domain_service,
    )

    oauth_application_repository = providers.Factory(
        SqlAlchemyOAuthApplicationRepository, session=core.request_context.provided.db_session
    )

    oauth_domain_service = providers.Factory(
        OAuthDomainService,
        repository=oauth_application_repository,
    )

    oauth_service = providers.Factory(
        OAuthService,
        domain_service=oauth_domain_service,
        user_service=user_center.user_service
    )


class PartnerAPIContainer(containers.DeclarativeContainer):

    core = providers.Container(Core)

    ai = providers.DependenciesContainer()

    roche_repository = providers.Factory(
        SQLAlchemyRocheRepository, session=core.request_context.provided.db_session
    )

    roche_domain_service = providers.Factory(
        RocheDomainService,
        repository=roche_repository,
    )

    roche_service = providers.Factory(
        RocheService,
        domain_service=roche_domain_service,
        ai_service=ai.ai_service
    )


class AppContainer(containers.DeclarativeContainer):

    core = providers.Container(Core)

    user_center = providers.Container(
        UserCenterContainer,
        core=core,
    )

    slice = providers.Container(
        SliceContainer,
        core=core,
        user_center=user_center
    )

    slice_analysis = providers.Container(
        SliceAnalysisContainer,
        core=core,
        slice=slice
    )

    ai = providers.Container(
        AIContainer,
        core=core,
        user_center=user_center,
        slice=slice,
        slice_analysis=slice_analysis
    )

    openapi = providers.Container(
        OpenAPIContainer,
        core=core,
        user_center=user_center
    )

    partner = providers.Container(
        PartnerAPIContainer,
        core=core,
        ai=ai
    )
