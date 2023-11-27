import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from cyborg.app.service_factory import AppServiceFactory
from cyborg.app.request_context import request_context
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.modules.user_center.utils.crypto import encrypt_password


def set_default_passwd():
    users = AppServiceFactory.user_service.domain_service.repository.get_users()
    for user in users:
        salt, encry_password = encrypt_password(user.password)
        user.update_data(
            salt=salt,
            encry_password=encry_password,
            password=''
        )
        AppServiceFactory.user_service.domain_service.repository.save(user)


def update_slice_is_marked():
    slices = AppServiceFactory.slice_service.domain_service.repository.get_slices(case_ids=None, per_page=sys.maxsize)
    for slice in slices:
        request_context.case_id = slice.caseid
        request_context.file_id = slice.fileid
        request_context.company = slice.company
        request_context.ai_type = AIType.get_by_value(slice.alg) or AIType.human
        is_marked = False
        try:
            is_marked = AppServiceFactory.new_slice_analysis_service().has_mark()
            if is_marked is False:
                continue
            AppServiceFactory.slice_service.domain_service.update_slice_is_marked(is_marked, slice)
        except Exception:
            print('update slice [%s] failed.' % slice.caseid)
            print(traceback.format_exc())
        print('update slice [%s] is_marked [%s] success.' % (slice.caseid, is_marked))


def update_company_ai_threshold():
    companies = AppServiceFactory.user_service.domain_service.company_repository.get_all_companies()
    for company in companies:
        if 'bm' in company.ai_threshold:
            continue
        ai_threshold = company.ai_threshold
        ai_threshold['bm'] = {"qc_cell_num": 500}
        company.update_data(
            ai_threshold=ai_threshold
        )
        AppServiceFactory.user_service.domain_service.company_repository.save(company)
    print('update company success')


def main():
    with request_context:
        update_company_ai_threshold()
        set_default_passwd()
        update_slice_is_marked()


if __name__ == '__main__':
    try:
        main()
        print('upgrade success.')
    except Exception as e:
        print('upgrade failed. errMsg=%s' % str(e))
