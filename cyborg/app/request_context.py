from contextvars import ContextVar

from sqlalchemy.orm import Session

from cyborg.app.auth import LoginUser
from cyborg.infra.session import get_session, get_session_by_db_uri


class RequestContext:
    _db_session: ContextVar[Session] = ContextVar("_db_session", default=None)
    _slice_db_session: ContextVar[Session] = ContextVar("_slice_db_session", default=None)
    _template_db_session: ContextVar[Session] = ContextVar("_template_db_session", default=None)
    _is_in_transaction: ContextVar[bool] = ContextVar("_is_in_transaction", default=False)
    _current_user: ContextVar = ContextVar("_current_user", default=None)
    _company: ContextVar = ContextVar("_company", default='')
    _case_id: ContextVar = ContextVar("_case_id", default='')
    _file_id: ContextVar = ContextVar("_file_id", default='')
    _ai_type: ContextVar = ContextVar("_ai_type", default=None)

    @property
    def db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._db_session

    @property
    def slice_db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._slice_db_session

    @property
    def template_db_session(self) -> ContextVar[Session]:
        return self._template_db_session

    @property
    def current_user(self) -> LoginUser:
        return self._current_user.get()

    @current_user.setter
    def current_user(self, user: LoginUser):
        self._current_user.set(user)

    @property
    def current_user_id(self) -> int:
        return self.current_user.id if self.current_user else 0

    @property
    def is_in_transaction(self):
        return self._is_in_transaction.get()

    @is_in_transaction.setter
    def is_in_transaction(self, value):
        self._is_in_transaction.set(value)

    @property
    def company(self):
        return self._company.get()

    @company.setter
    def company(self, value):
        self._company.set(value)

    @property
    def current_company(self):
        return self.current_user.company if self.current_user else self.company

    @property
    def case_id(self):
        return self._case_id.get()

    @case_id.setter
    def case_id(self, value):
        self._case_id.set(value)

    @property
    def file_id(self):
        return self._file_id.get()

    @file_id.setter
    def file_id(self, value):
        self._file_id.set(value)

    @property
    def ai_type(self):
        return self._ai_type.get()

    @ai_type.setter
    def ai_type(self, value):
        self._ai_type.set(value)

    def begin_request(self):
        session = get_session()
        session.begin()
        self._db_session.set(session)

    def end_request(self, commit=False):
        session = self.db_session.get()
        if session:
            if commit:
                session.commit()
            else:
                session.rollback()

    def connect_slice_db(self, db_file_path: str):
        session = get_session_by_db_uri(f'sqlite:///{db_file_path}')
        session.begin()
        self.slice_db_session.set(session)

    def connect_template_db(self, db_file_path: str):
        session = get_session_by_db_uri(f'sqlite:///{db_file_path}')
        session.begin()
        self.template_db_session.set(session)

    def close_slice_db(self, commit=True):
        session = self.slice_db_session.get()
        if session:
            if commit:
                session.commit()
            else:
                session.rollback()
            session.close()

        temp_session = self.template_db_session.get()
        if temp_session:
            temp_session.rollback()
            temp_session.close()

    def __enter__(self):
        self.begin_request()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        commit = exc_type is None
        self.end_request(commit=commit)


request_context = RequestContext()
