import datetime
import json
import logging
import os
import signal
from collections import Counter
from typing import Optional, List, Type, Tuple

import cv2
import numpy as np
import psutil
from celery.result import AsyncResult
from celery.exceptions import TimeoutError as CeleryTimeoutError

from cyborg.app.settings import Settings
from cyborg.celery.app import app as celery_app
from cyborg.consts.bm import BMConsts
from cyborg.consts.common import Consts
from cyborg.consts.her2 import Her2Consts
from cyborg.consts.ki67 import Ki67Consts
from cyborg.consts.np import NPConsts
from cyborg.consts.pdl1 import Pdl1Consts
from cyborg.consts.tct import TCTConsts
from cyborg.infra.cache import cache
from cyborg.infra.fs import fs
from cyborg.infra.redlock import with_redlock
from cyborg.infra.session import transaction
from cyborg.libs.heimdall.dispatch import open_slide
from cyborg.modules.ai.domain.entities import AITaskEntity, AIStatisticsEntity, TCTProbEntity
from cyborg.modules.ai.domain.repositories import AIRepository
from cyborg.modules.ai.domain.value_objects import ALGResult, Mark, AITaskStatus, TCTDiagnosisType, MicrobeType
from cyborg.seedwork.domain.value_objects import AIType
from cyborg.utils.id_worker import IdWorker

logger = logging.getLogger(__name__)


class AIDomainService(object):
    RANK0_TASK_ID_CACHE_KEY = 'cyborg:ai_task:rank0'

    RUNNING_AI_TASKS_CACHE_KEY = 'cyborg:running_ai_tasks'

    CELERY_WORKER_PID_CACHE_KEY = 'celery_worker_pid:{0}'

    def __init__(self, repository: AIRepository):
        super(AIDomainService, self).__init__()
        self.repository = repository

    def create_ai_task(
            self,
            ai_type: AIType,
            case_id: str,
            file_id: str,
            rois: Optional[list] = None,
            model_info: Optional[dict] = None,
            template_id: Optional[int] = None,
            is_calibrate: bool = False
    ) -> Optional[AITaskEntity]:

        task = AITaskEntity(raw_data={
            'ai_type': ai_type,
            'case_id': case_id,
            'file_id': file_id,
            'rois': rois,
            'model_info': model_info,
            'template_id': template_id,
            'is_calibrate': is_calibrate
        })

        if self.repository.save_ai_task(task):
            return task

        return None

    def update_ai_task(
            self, task: AITaskEntity, status: Optional[AITaskStatus] = None, result_id: Optional[str] = None
    ) -> bool:

        if status is not None:
            task.update_data(status=status)
            if status == AITaskStatus.analyzing:
                task.setup_expired_time()
                cache.set(self.RANK0_TASK_ID_CACHE_KEY, task.id)

        if result_id is not None:
            task.update_data(result_id=result_id)

        return self.repository.save_ai_task(task)

    def maintain_ai_tasks(self, until_id: Optional[int] = None) -> List[dict]:
        tasks = self.repository.get_ai_tasks(status=AITaskStatus.analyzing, until_id=until_id, limit=100)
        failed = []
        for task in tasks:
            if task.is_timeout:
                task.set_failed()
                self.repository.save_ai_task(task)
                failed.append(task.to_dict())
        return failed

    def reset_running_tasks(self, purge_ranking: bool = False) -> int:
        tasks = self.repository.get_ai_tasks(status=AITaskStatus.analyzing)
        purged = 0
        for task in tasks:
            task.reset()
            if self.repository.save_ai_task(task):
                purged += 1

        if purge_ranking:
            tasks = self.repository.get_ai_tasks(status=AITaskStatus.default)
            for task in tasks:
                self.repository.delete_ai_task(task.id)

        cache.delete(self.RUNNING_AI_TASKS_CACHE_KEY)
        return purged

    def check_available_gpu(self, ai_type: AIType, slide_path: str) -> List[str]:

        required_gpu_num, required_gpu_memory = Consts.MODEL_SIZE.get(ai_type, (1, 10))

        from cyborg.modules.ai.utils.gpu import get_gpu_status
        gpu_status = get_gpu_status()
        if isinstance(required_gpu_num, tuple):
            min_gpu_num, max_gpu_num = required_gpu_num
        else:
            min_gpu_num, max_gpu_num = required_gpu_num, required_gpu_num

        # single gpu for small images is enough
        if isinstance(slide_path, str) and fs.path_splitext(slide_path.lower())[1] in ['.jpg', '.png', '.jpeg', '.bmp']:
            min_gpu_num, max_gpu_num = 1, 1

        selected_gpus = []
        for gpu_id, memory in gpu_status.items():
            memory = gpu_status[gpu_id]
            if memory['free'] >= required_gpu_memory << 10 and gpu_id not in selected_gpus:
                selected_gpus.append(gpu_id)
                if len(selected_gpus) >= max_gpu_num:
                    return selected_gpus
        if len(selected_gpus) >= min_gpu_num:
            return selected_gpus
        else:
            return []

    @with_redlock('mark_ai_task_running')
    def mark_ai_task_running(self, ai_task: AITaskEntity) -> bool:
        cache_key = self.RUNNING_AI_TASKS_CACHE_KEY
        ai_task_ids = cache.smembers(cache_key)
        total_gpu_mem = 0
        for ai_task_id in ai_task_ids:
            _ai_task = self.repository.get_ai_task_by_id(ai_task_id)
            if _ai_task:
                _, current_gpu_memory = Consts.MODEL_SIZE.get(_ai_task.ai_type, (1, 10))
                total_gpu_mem += current_gpu_memory

        _, required_gpu_memory = Consts.MODEL_SIZE.get(ai_task.ai_type, (1, 10))

        if total_gpu_mem + required_gpu_memory >= Settings.TOTAL_GPU_MEM:
            return False

        cache.sadd(cache_key, ai_task.id)
        return True

    def unmark_ai_task_running(self, ai_task: AITaskEntity):
        return cache.srem(self.RUNNING_AI_TASKS_CACHE_KEY, ai_task.id)

    def kill_task_processes(self, pid: int) -> bool:
        p = psutil.Process(pid)
        for process in p.children(recursive=True):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception as e:
                logger.warning(e)

        p.terminate()
        return True

    def get_ai_statistics(self, ai_type: AIType, company: str, start_date: str, end_date: str) -> List[dict]:
        stats_list = self.repository.get_ai_stats(
            ai_type=ai_type, company=company, start_date=start_date, end_date=end_date, version=Settings.VERSION)
        data = [stats.to_dict() for stats in stats_list]
        total = AIStatisticsEntity().to_stats_data()
        total.update(dict(sum([Counter(stats.to_stats_data()) for stats in stats_list], start=Counter())))
        total['date'] = '总计'
        data.insert(0, total)
        return data

    def refresh_ai_statistics(
            self, ai_suggest: str, ai_type: AIType, slice_info: dict, is_error: bool
    ) -> Optional[AIStatisticsEntity]:
        """
        刷新ai计算统计数据
        """
        company = slice_info['company']
        current_stats_id = slice_info['as_id']
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        stats_list = self.repository.get_ai_stats(ai_type=ai_type, company=company, date=current_date)
        stats = stats_list[0] if stats_list else None
        if not stats:
            stats = AIStatisticsEntity(raw_data=dict(
                date=current_date,
                total_count=0,
                negative_count=0,
                positive_count=0,
                abnormal_count=0,
                total_count_dr=0,
                negative_count_dr=0,
                positive_count_dr=0,
                abnormal_count_dr=0,
                company=company,
                ai_type=ai_type,
                version=Settings.VERSION
            ))
        if not self.repository.save_ai_stats(stats):
            return None

        stats.update_data(total_count=stats.total_count + 1)
        if stats.id != current_stats_id:
            stats.update_data(total_count_dr=stats.total_count_dr + 1)

        if is_error:
            stats.update_data(abnormal_count=stats.abnormal_count + 1)
            if stats.id != current_stats_id:
                stats.update_data(abnormal_count_dr=stats.abnormal_count_dr + 1)
        else:
            if ai_suggest:
                if ai_suggest.startswith('阳性'):
                    stats.update_data(positive_count=stats.positive_count + 1)
                    if stats.id != current_stats_id:
                        stats.update_data(positive_count_dr=stats.positive_count_dr + 1)
                elif ai_suggest.startswith('阴性'):
                    stats.update_data(negative_count=stats.negative_count + 1)
                    if stats.id != current_stats_id:
                        stats.update_data(negative_count_dr=stats.negative_count_dr + 1)

        if self.repository.save_ai_stats(stats):
            return stats
        return None

    def select_alg(
            self, ai_type: AIType, model_type: Optional[str] = None, model_name: Optional[str] = None
    ) -> Optional[Type]:
        from cyborg.modules.ai.libs.algorithms.TCTAnalysis_v2_2.tct_alg import (
            LCT_mobile_micro0324, LCT40k_convnext_nofz, LCT40k_convnext_HDX, LCT_mix80k0417_8)
        from cyborg.modules.ai.libs.algorithms.TCTAnalysis_v3_1.tct_alg import TCT_ALG2
        if ai_type in [AIType.tct, AIType.lct, AIType.dna]:
            models = {
                "1": LCT_mobile_micro0324,
                "2": TCT_ALG2,
                "LCT40k_convnext_nofz": LCT40k_convnext_nofz,
                "LCT40k_convnext_HDX": LCT40k_convnext_HDX,
                "LCT_mobile_micro0324": LCT_mobile_micro0324,
                "LCT_mix80k0417_8": LCT_mix80k0417_8,
            }
            return models.get(model_name, models.get(model_type, LCT_mobile_micro0324))
        return None

    def save_prob(self, slice_id: int, prob_info: dict) -> bool:
        tct_prob = self.repository.get_tct_prob(slice_id=slice_id)
        if not tct_prob:
            tct_prob = TCTProbEntity()
            tct_prob.update_data(slice_id=slice_id)

        tct_prob.update_data(
            prob_nilm=prob_info['NILM'],
            prob_ascus=prob_info['ASC-US'],
            prob_lsil=prob_info['LSIL'],
            prob_asch=prob_info['ASC-H'],
            prob_agc=prob_info['AGC'],
            prob_hsil=prob_info['HSIL'],
        )

        return self.repository.save_tct_prob(tct_prob)

    def run_tct(self, task: AITaskEntity) -> ALGResult:
        model_info = task.model_info
        threshold = model_info.get('ai_threshold')
        model_type = model_info.get('model_type')
        model_name = model_info.get('model_name')

        alg_class = self.select_alg(task.ai_type, model_type, model_name)

        slide = open_slide(task.slide_path)

        roi_marks = []
        prob_dict = None
        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            if alg_class.__name__ == 'TCT_ALG2':
                config_path = task.ai_type.ai_name + model_type if model_type.isdigit() else model_type
                alg_obj = alg_class(config_path=config_path, threshold=threshold)
                result = alg_obj.cal_tct(slide)

                from cyborg.modules.ai.utils.tct import generate_ai_result2
                ai_result = generate_ai_result2(result=result, roiid=roi['id'])
            else:
                alg_obj = alg_class(threshold=threshold)
                result = alg_obj.cal_tct(slide)

                from cyborg.modules.ai.utils.tct import generate_ai_result
                ai_result = generate_ai_result(result=result, roiid=roi['id'])

            from cyborg.modules.ai.utils.prob import save_prob_to_file
            prob_dict = save_prob_to_file(slide_path=task.slide_path, result=result)

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                method='rectangle',
                mark_type=3,
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result
            ))

        ai_result = roi_marks[0].ai_result

        return ALGResult(
            roi_marks=roi_marks,
            ai_suggest=' '.join(ai_result['diagnosis']) + ' ' + ','.join(ai_result['microbe']),
            slide_quality=ai_result['slide_quality'],
            cell_num=ai_result['cell_num'],
            prob_dict=prob_dict
        )

    def run_lct(self, task: AITaskEntity) -> ALGResult:
        return self.run_tct(task)

    def run_tbs_dna(self, task: AITaskEntity) -> ALGResult:
        model_info = task.model_info
        threshold = model_info.get('ai_threshold')
        model_type = model_info.get('model_type')

        alg_class = self.select_alg(task.ai_type, model_type)

        from cyborg.modules.ai.libs.algorithms.DNA1.dna_alg import DNA_1020
        dna_alg_class = DNA_1020

        slide = open_slide(task.slide_path)

        roi_marks = []
        prob_dict = None

        from cyborg.modules.ai.utils.prob import save_prob_to_file
        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            tbs_result = alg_class(threshold).cal_tct(slide)
            dna_result = dna_alg_class().dna_test(slide)

            prob_dict = save_prob_to_file(slide_path=task.slide_path, result=tbs_result)

            from cyborg.modules.ai.utils.tct import generate_ai_result, generate_dna_ai_result
            ai_result = generate_ai_result(result=tbs_result, roiid=roi['id'])
            ai_result.update(generate_dna_ai_result(result=dna_result, roiid=roi['id']))

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                mark_type=3,
                method='rectangle',
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result,
            ))

        ai_result = roi_marks[0].ai_result

        ai_suggest = f"{' '.join(ai_result['diagnosis'])} {','.join(ai_result['microbe'])};{ai_result['dna_diagnosis']}"
        return ALGResult(
            ai_suggest=ai_suggest,
            roi_marks=roi_marks,
            slide_quality=ai_result['slide_quality'],
            cell_num=ai_result['cell_num'],
            prob_dict=prob_dict
        )

    def run_dna_ploidy(self, task: AITaskEntity) -> ALGResult:
        model_info = task.model_info
        threshold = model_info.get('ai_threshold')
        if isinstance(threshold, dict):
            iod_ratio = threshold.get("iod_ratio", 0.25)
            conf_thres = threshold.get("conf_thres", 0.42)
        else:
            iod_ratio = 0.25
            conf_thres = 0.42

        from cyborg.modules.ai.libs.algorithms.DNA2.dna_alg import DNA_1020
        dna_alg_class = DNA_1020

        slide = open_slide(task.slide_path)

        roi_marks = []
        prob_dict = None

        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            dna_ploidy_result = dna_alg_class(iod_ratio=iod_ratio, conf_thres=conf_thres).dna_test(slide)

            from cyborg.modules.ai.utils.tct import generate_dna_ploidy_aiResult
            ai_result = generate_dna_ploidy_aiResult(result=dna_ploidy_result, roiid=roi['id'])

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                mark_type=3,
                method='rectangle',
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result,
            ))

        ai_result = roi_marks[0].ai_result

        return ALGResult(
            ai_suggest=ai_result['dna_diagnosis'],
            roi_marks=roi_marks,
            cell_num=ai_result['cell_num'],
            prob_dict=prob_dict
        )

    def run_her2(self, task: AITaskEntity, group_name_to_id: dict):
        cell_marks = []
        roi_marks = []
        ai_result = {}

        slide = open_slide(task.slide_path)
        mpp = slide.mpp or 0.242042

        rois = task.rois or [task.new_default_roi()]

        from cyborg.modules.ai.libs.algorithms.Her2New_.detect_all import run_her2_alg, roi_filter

        center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = run_her2_alg(
            slide_path=task.slide_path, roi_list=rois)

        for roi in rois:
            roi_id, x_coords, y_coords = roi['id'], roi['x'], roi['y']
            center_coords, cls_labels = roi_filter(
                center_coords_np_with_id[roi_id],
                cls_labels_np_with_id[roi_id],
                x_coords,
                y_coords
            )

            ai_result = Her2Consts.rois_summary_dict.copy()
            label_to_roi_name = Her2Consts.cell_label_dict
            for idx, coord in enumerate(center_coords):
                roi_name = label_to_roi_name[str(cls_labels[idx])]
                ai_result[roi_name] += 1

                x = float(coord[0]) if slide.width > float(coord[0]) else float(slide.width - 1)
                y = float(coord[1]) if slide.height > float(coord[1]) else float(slide.height - 1)

                mark = Mark(
                    position={'x': [x], 'y': [y]},
                    fill_color=Her2Consts.type_color_dict[Her2Consts.label_to_diagnosis_type[int(cls_labels[idx])]],
                    mark_type=2,
                    diagnosis={'type': Her2Consts.label_to_diagnosis_type[int(cls_labels[idx])]},
                    radius=1 / mpp,
                    editable=0,
                    group_id=group_name_to_id.get(Her2Consts.idx_to_label[int(cls_labels[idx])]),
                    area_id=roi_id,
                    method='spot'
                )
                cell_marks.append(mark)

            whole_slide = 1 if len(x_coords) == 0 else 0
            group_id = group_name_to_id.get('ROI') if whole_slide != 1 else None

            ai_result.update({
                'whole_slide': whole_slide,
                '分级结果': Her2Consts.level[int(lvl)]
            })

            roi_marks.append(Mark(
                id=roi_id,
                position={'x': x_coords, 'y': y_coords},
                method='rectangle',
                mark_type=3,
                is_export=1,
                ai_result=ai_result,
                editable=1,
                stroke_color='grey',
                group_id=group_id
            ))

        return ALGResult(
            ai_suggest=ai_result['分级结果'],
            cell_marks=cell_marks,
            roi_marks=roi_marks,
        )

    def run_pdl1(self, task: AITaskEntity, group_name_to_id: dict, fitting_data_dir: str):
        all_center_coords_list = []
        all_cell_types_list = []
        all_ori_cell_types_list = []
        all_probs_list = []
        slide = open_slide(task.slide_path)
        mpp = slide.mpp if slide.mpp is not None else 0.242042

        xcent = int(slide.width / 2)
        ycent = int(slide.height / 2)
        total_pos_tumor_num, total_neg_tumor_num = 0, 0

        from cyborg.modules.ai.utils.pdl1 import fitting_target_tps_update, compute_pdl1_s
        if fs.path_exists(fs.path_join(fitting_data_dir, "fittingCurve.xlsx")) and fs.path_exists(
                fs.path_join(fitting_data_dir, "smooth.json")):
            excel_path = fs.path_join(fitting_data_dir, "fittingCurve.xlsx")
            smooth = open(fs.path_join(fitting_data_dir, "smooth.json"), 'r')
            smooth = float(json.loads(smooth.read())["smooth_value"])
            fitting_model = fitting_target_tps_update(excel_path=excel_path, smooth_value=smooth)
        else:
            fitting_model = None
            smooth = None

        roi_marks, cell_marks = [], []

        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            roi_id, x_coords, y_coords = roi['id'], roi['x'], roi['y']

            whole_slide = 1 if len(x_coords) == 0 else 0

            count_summary_dict, center_coords, ori_labels, cls_labels, probs, annot_cls_labels = compute_pdl1_s(
                slide_path=task.slide_path,
                x_coords=x_coords,
                y_coords=y_coords, fitting_model=fitting_model, smooth=smooth)

            total_pos_tumor_num += count_summary_dict['pos_tumor']
            total_neg_tumor_num += count_summary_dict['neg_tumor']

            count_summary_dict['whole_slide'] = whole_slide
            all_center_coords_list += center_coords
            all_cell_types_list += cls_labels
            all_ori_cell_types_list += ori_labels
            all_probs_list += probs
            # split into 4 region
            if whole_slide:
                if len(center_coords) > 0:
                    center_coords_np, cls_labels_np, _ = np.array(center_coords), np.array(
                        cls_labels), np.array(probs)
                    p1_index = np.where(np.logical_and(
                        center_coords_np[:, 0] < xcent, center_coords_np[:, 1] < ycent))[0]
                    p2_index = np.where(
                        np.logical_and(center_coords_np[:, 0] > xcent, center_coords_np[:, 1] < ycent))[0]
                    p3_index = np.where(
                        np.logical_and(center_coords_np[:, 0] < xcent, center_coords_np[:, 1] > ycent))[0]
                    p4_index = np.where(
                        np.logical_and(center_coords_np[:, 0] > xcent, center_coords_np[:, 1] > ycent))[0]
                    for i, region_idx in enumerate([p1_index, p2_index, p3_index, p4_index]):
                        cells = cls_labels_np[region_idx]
                        counter = Counter(cells)
                        neg_norm = counter[Pdl1Consts.cell_label_dict['neg_norm']]
                        neg_tumor = counter[Pdl1Consts.cell_label_dict['neg_tumor']]
                        pos_norm = counter[Pdl1Consts.cell_label_dict['pos_norm']]
                        pos_tumor = counter[Pdl1Consts.cell_label_dict['pos_tumor']]
                        total = neg_tumor + neg_norm + pos_tumor + pos_norm
                        tps = round(float(pos_tumor / (pos_tumor + neg_tumor + 1e-10)), 4)
                        count_summary_dict.update({
                            str(i + 1): {
                                'pos_tumor': pos_tumor, 'neg_tumor': neg_tumor, 'pos_norm': pos_norm,
                                'neg_norm': neg_norm, 'total': total, 'tps': tps
                            }
                        })
                else:
                    temp_dict = {'neg_norm': 0, 'neg_tumor': 0, 'pos_norm': 0, 'pos_tumor': 0, 'total': 0,
                                 'tps': 0}
                    count_summary_dict.update({"1": temp_dict, "2": temp_dict, "3": temp_dict, "4": temp_dict})

            for center_coords, cell_type, annot_type in zip(center_coords, cls_labels, annot_cls_labels):
                mark = Mark(
                    position={'x': [center_coords[0]], 'y': [center_coords[1]]},
                    fill_color=Pdl1Consts.display_color_dict[cell_type],
                    mark_type=2,
                    method='spot',
                    diagnosis={'type': cell_type},
                    radius=1 / mpp,
                    area_id=roi_id,
                    editable=0,
                    group_id=group_name_to_id.get(Pdl1Consts.reversed_annot_clss_map_dict[annot_type]),
                )
                cell_marks.append(mark)

            count_summary_dict['center_coords'] = [xcent, ycent]

            roi_marks.append(Mark(
                id=roi_id,
                position={'x': x_coords, 'y': y_coords},
                mark_type=3,
                method='rectangle',
                is_export=1,
                stroke_color='grey',
                ai_result=count_summary_dict,
                group_id=group_name_to_id.get('ROI')
            ))

        tps = str(
            round(100 * total_pos_tumor_num / (total_neg_tumor_num + total_pos_tumor_num + 1e-10), 2)) + '%'

        return ALGResult(
            ai_suggest=tps,
            roi_marks=roi_marks,
            cell_marks=cell_marks,
        )

    def run_np(self, task: AITaskEntity, group_name_to_id: dict) -> ALGResult:
        slide = open_slide(task.slide_path)
        mpp = slide.mpp or 0.242042
        xcent = int(slide.width / 2)
        ycent = int(slide.height / 2)

        cell_marks = []
        roi_marks = []
        all_cell_center_coords = []
        all_roi_coords = []
        all_cell_types = []

        whole_slide = 0

        for roi in task.rois or [task.new_default_roi()]:
            roi_id, x_coords, y_coords = roi['id'], roi['x'], roi['y']
            if len(x_coords) == 0:
                whole_slide = 1

            from cyborg.modules.ai.libs.algorithms.NP.run_np import cal_np
            cell_coords, cell_labels, contour_coords, contour_labels, total_area = cal_np(
                slide_path=task.slide_path, x_coords=x_coords, y_coords=y_coords)

            all_cell_center_coords += list(cell_coords)
            all_roi_coords += list(contour_coords)
            all_cell_types += list(cell_labels)

            ai_result = {'whole_slide': whole_slide, 'center_coords': [xcent, ycent]}
            cell_counter = Counter(cell_labels)
            contour_counter = Counter(contour_labels)

            if whole_slide:
                if len(cell_coords) > 0:
                    center_coords_np, cls_labels_np = np.array(cell_coords), np.array(cell_labels)
                    p1_index = np.where(
                        np.logical_and(center_coords_np[:, 0] < xcent, center_coords_np[:, 1] < ycent))[0]
                    p2_index = np.where(
                        np.logical_and(center_coords_np[:, 0] > xcent, center_coords_np[:, 1] < ycent))[0]
                    p3_index = np.where(
                        np.logical_and(center_coords_np[:, 0] < xcent, center_coords_np[:, 1] > ycent))[0]
                    p4_index = np.where(
                        np.logical_and(center_coords_np[:, 0] > xcent, center_coords_np[:, 1] > ycent))[0]

                    for idx, region_idx in enumerate([p1_index, p2_index, p3_index, p4_index]):
                        cells = cls_labels_np[region_idx]
                        counter = Counter(cells)
                        count_dict = {'total': 0}
                        for k, v in NPConsts.cell_label_dict.items():
                            count_dict[k] = counter[v]
                            count_dict['total'] += counter[v]
                        ai_result.update({
                            str(idx + 1): count_dict})
                else:
                    temp_dict = {k: 0 for k in NPConsts.cell_label_dict.keys()}
                    temp_dict['total'] = 0
                    ai_result.update({"1": temp_dict, "2": temp_dict, "3": temp_dict, "4": temp_dict})

            for k, v in NPConsts.cell_label_dict.items():
                ai_result[k] = {
                    'count': cell_counter[v],
                    'index': round(float(cell_counter[v] / max(1, len(cell_labels))) * 100, 2),
                    'area': None
                }

            for k, v in NPConsts.roi_label_dict.items():
                this_type_idx = np.where(np.array(contour_labels) == v)[0]
                area = 0
                for idx in this_type_idx:
                    area += cv2.contourArea(np.array(contour_coords[idx]).astype(np.int32)) * mpp ** 2
                ai_result[k] = {
                    'count': contour_counter[v],
                    'index': round(area / max(1, total_area * mpp ** 2) * 100, 2),
                    'area': round(area, 2),
                    'total_area': round(total_area * mpp ** 2, 2)
                }

            roi_marks.append(Mark(
                id=roi_id,
                position={'x': x_coords, 'y': y_coords},
                mark_type=3,
                method='rectangle',
                ai_result=ai_result,
                stroke_color='grey',
                is_export=1,
                group_id=group_name_to_id.get('ROI'),
            ))

            for coord, label in zip(list(cell_coords), list(cell_labels)):
                cell_mark = Mark(
                    position={'x': [int(coord[0])], 'y': [int(coord[1])]},
                    fill_color=NPConsts.display_color_dict[NPConsts.reversed_cell_label_dict[label]],
                    mark_type=2,
                    diagnosis={'type': NPConsts.return_diagnosis_type[NPConsts.reversed_cell_label_dict[label]]},
                    radius=1 / mpp,
                    area_id=roi_id,
                    editable=1,
                    group_id=group_name_to_id.get(NPConsts.reversed_cell_label_dict[label]),
                    method='spot'
                )
                cell_marks.append(cell_mark)

            for contour_coord, contour_label in zip(list(contour_coords), list(contour_labels)):
                roi_mark = Mark(
                    position={
                        'x': list(map(int, np.array(contour_coord)[:, 0])),
                        'y': list(map(int, np.array(contour_coord)[:, 1]))
                    },
                    stroke_color=NPConsts.display_color_dict[NPConsts.reversed_roi_label_dict[contour_label]],
                    mark_type=2,
                    method='freepen',
                    diagnosis={'type': NPConsts.return_diagnosis_type[NPConsts.reversed_roi_label_dict[contour_label]]},
                    radius=1 / mpp,
                    area_id=roi_id,
                    editable=1,
                    group_id=group_name_to_id.get(NPConsts.reversed_roi_label_dict[contour_label])
                )
                roi_marks.append(roi_mark)

        all_cell_types_counter = Counter(all_cell_types)

        if len(all_cell_types) == 0:
            return ALGResult(ai_suggest='无细胞检出')

        diagnosis_name = NPConsts.reversed_cell_label_dict[all_cell_types_counter.most_common(1)[0][0]]
        percent = str(round(float(all_cell_types_counter.most_common(1)[0][1] / max(1, len(all_cell_types)) * 100), 2))
        ai_suggest = f'{diagnosis_name}: {percent}%'

        return ALGResult(
            ai_suggest=ai_suggest,
            cell_marks=cell_marks,
            roi_marks=roi_marks
        )

    def run_ki67(self, task: AITaskEntity, group_name_to_id: dict, compute_wsi: bool = False):

        all_center_coords_list = []

        slide = open_slide(task.slide_path)
        mpp = slide.mpp if slide.mpp is not None else 0.242042

        whole_slide = 1 if compute_wsi else 0
        pos_num = 0
        total_cell_num = 0
        total_normal_cell = 0

        logger.info(f'compute_wsi: {compute_wsi}')

        cell_marks = []
        roi_marks = []

        roi_list = task.rois
        if not roi_list and compute_wsi:
            roi_list = [task.new_default_roi()]

        from cyborg.modules.ai.libs.algorithms.Ki67.main import WSITester
        result_df = WSITester().run(slide, compute_wsi=compute_wsi, roi_list=roi_list)
        id_worker = IdWorker.new_mark_id_worker()

        for idx in range(len(result_df)):
            item = result_df.iloc[idx]

            roi_id = int(item.get('id', 0))
            if roi_id <= 0:
                roi_id = id_worker.new_mark_id_worker().get_new_id()

            x_coords, y_coords = list(map(int, item['x_coords'])), list(map(int, item['y_coords']))
            roi_type = item['类别'] if not roi_id else 4
            center_coords, cls_labels = (item['检测坐标'][:, :2]), item['检测坐标'][:, 2]

            if len(center_coords) > 0:
                center_coords = center_coords.tolist()
                remap_cls_labels = np.vectorize(Ki67Consts.label_to_diagnosis_type.get)(cls_labels).tolist()
                all_center_coords_list += center_coords
            else:
                center_coords, cls_labels, remap_cls_labels = [], [], []

            for center_coord, cell_type, remap_cell_type in zip(center_coords, cls_labels, remap_cls_labels):
                cell_mark = Mark(
                    position={'x': [center_coord[0]], 'y': [center_coord[1]]},
                    fill_color=Ki67Consts.cell_color_dict[remap_cell_type],
                    mark_type=2,
                    method='spot',
                    diagnosis={'type': remap_cell_type},
                    radius=1 / mpp,
                    area_id=roi_id,
                    group_id=group_name_to_id.get(Ki67Consts.reversed_cell_label_dict[cell_type])
                )
                cell_marks.append(cell_mark)

            pos_num += item['阳性肿瘤']
            total_cell_num += item['肿瘤细胞']
            total_normal_cell += int(np.sum(np.array(remap_cls_labels) == Ki67Consts.display_cell_dict['非肿瘤']))

            if not compute_wsi:
                response_count_dict = {
                    'total': int(item['肿瘤细胞']),
                    'pos_tumor': int(item['阳性肿瘤']),
                    'neg_tumor': int(item['阴性肿瘤']),
                    'normal_cell': int(np.sum(np.array(remap_cls_labels) == Ki67Consts.display_cell_dict['非肿瘤'])),
                    'index': float(item['ki67指数']),
                    'whole_slide': whole_slide
                }
                roi_marks.append(Mark(
                    id=roi_id,
                    position={'x': x_coords, 'y': y_coords},
                    mark_type=3,
                    method='rectangle',
                    stroke_color='grey',
                    radius=5,
                    ai_result=response_count_dict,
                    editable=1,
                    is_export=1,
                    group_id=group_name_to_id.get(Ki67Consts.reversed_roi_label_dict[roi_type])
                ))

        if compute_wsi:
            response_count_dict = {
                'total': int(total_cell_num),
                'pos_tumor': int(pos_num),
                'neg_tumor': int(total_cell_num) - int(pos_num),
                'normal_cell': int(total_normal_cell),
                'index': round(float(pos_num / (total_cell_num + 1e-4)), 4),
                'whole_slide': whole_slide
            }
            whole_slide_roi = roi_list[0]
            roi_marks.append(Mark(
                id=whole_slide_roi['id'],
                position={'x': [], 'y': []},
                mark_type=3,
                method='rectangle',
                stroke_color='grey',
                radius=5,
                ai_result=response_count_dict,
                editable=1,
                is_export=1,
                group_id=group_name_to_id.get('ROI')
            ))

        return ALGResult(
            ai_suggest=str(round(float(pos_num / (total_cell_num + 1e-4)) * 100, 2)) + '%',
            cell_marks=cell_marks,
            roi_marks=roi_marks
        )

    def run_ki67_new(self, task: AITaskEntity, group_name_to_id: dict):
        slide = open_slide(task.slide_path)
        mpp = slide.mpp if slide.mpp is not None else 0.242042
        total_pos_tumor_num, total_neg_tumor_num = 0, 0
        xcent = int(slide.width / 2)
        ycent = int(slide.height / 2)
        all_center_coords_list = []
        all_cell_types_list = []
        all_ori_cell_types_list = []
        all_probs_list = []
        roi_marks, cell_marks = [], []

        from cyborg.modules.ai.libs.algorithms.Ki67New.run_ki67 import cal_ki67
        from cyborg.consts.ki67new import Ki67NewConsts

        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            roi_id, x_coords, y_coords = roi['id'], roi['x'], roi['y']
            whole_slide = 1 if len(x_coords) == 0 else 0

            center_coords_np, cls_labels_np, probs_np = cal_ki67(task.slide_path, x_coords=x_coords, y_coords=y_coords)
            changed_cls_labels = cls_labels_np
            remap_changed_cls_labels = np.vectorize(Ki67NewConsts.map_dict.get)(changed_cls_labels)
            remap_ori_cls_labels = np.vectorize(Ki67NewConsts.map_dict.get)(cls_labels_np)
            if remap_changed_cls_labels is not None:
                cell_count = Counter(remap_changed_cls_labels)
                count_summary_dict = {'pos_tumor': int(cell_count[Ki67NewConsts.annot_clss_map_dict['阳性肿瘤细胞']]),
                                      'neg_tumor': int(cell_count[Ki67NewConsts.annot_clss_map_dict['阴性肿瘤细胞']]),
                                      'normal_cell': int(cell_count[Ki67NewConsts.annot_clss_map_dict['阳性组织细胞']] +
                                                         cell_count[Ki67NewConsts.annot_clss_map_dict['阴性组织细胞']]),
                                      'total': int(cell_count[Ki67NewConsts.annot_clss_map_dict['阳性肿瘤细胞']] +
                                                   cell_count[Ki67NewConsts.annot_clss_map_dict['阴性肿瘤细胞']]),
                                      'index': round(
                                          float(cell_count[Ki67NewConsts.annot_clss_map_dict['阳性肿瘤细胞']] /
                                                (cell_count[Ki67NewConsts.annot_clss_map_dict['阳性肿瘤细胞']] +
                                                 cell_count[Ki67NewConsts.annot_clss_map_dict['阴性肿瘤细胞']] + 1e-8)),
                                          4)}
                center_coords = center_coords_np.tolist()
                cls_labels = remap_changed_cls_labels.tolist()
                probs = probs_np.tolist()
                annot_cls_labels = changed_cls_labels.tolist()
                ori_labels = remap_ori_cls_labels.tolist()

                total_pos_tumor_num += count_summary_dict['pos_tumor']
                total_neg_tumor_num += count_summary_dict['neg_tumor']

                count_summary_dict['whole_slide'] = whole_slide

                all_center_coords_list += center_coords
                all_cell_types_list += cls_labels
                all_ori_cell_types_list += ori_labels
                all_probs_list += probs

                for center_coords, cell_type, annot_type in zip(center_coords, cls_labels, annot_cls_labels):
                    mark = Mark(
                        position={'x': [center_coords[0]], 'y': [center_coords[1]]},
                        fill_color=Ki67NewConsts.type_color_dict[cell_type],
                        mark_type=2,
                        method='spot',
                        diagnosis={'type': cell_type},
                        radius=1 / mpp,
                        area_id=roi_id,
                        editable=0,
                        group_id=group_name_to_id.get(Ki67NewConsts.reversed_annot_clss_map_dict[annot_type]),
                    )
                    cell_marks.append(mark)

                count_summary_dict['center_coords'] = [xcent, ycent]

                roi_marks.append(Mark(
                    id=roi_id,
                    position={'x': x_coords, 'y': y_coords},
                    mark_type=3,
                    method='rectangle',
                    is_export=1,
                    stroke_color='grey',
                    ai_result=count_summary_dict,
                    group_id=group_name_to_id.get('ROI')
                ))

        if total_pos_tumor_num + total_neg_tumor_num > 100:
            ki67_index = round(total_pos_tumor_num / (total_neg_tumor_num + total_pos_tumor_num) * 100, 2)
            return_value = f'{ki67_index}%'
        else:
            return_value = '肿瘤细胞检出不足'

        return ALGResult(
            ai_suggest=return_value,
            roi_marks=roi_marks,
            cell_marks=cell_marks,
        )

    def run_fish_tissue(self, task: AITaskEntity, group_name_to_id: dict):
        nucleus_group_id = group_name_to_id['可计数细胞核(组织)']
        red_group_id = group_name_to_id['红色信号点']
        green_group_id = group_name_to_id['绿色信号点']

        type_color_dict = {0: 'white', 1: 'red', 2: '#00ff15'}
        mpp = 0.5

        whole_slide = 1
        rois = task.rois or [task.new_default_roi()]
        roi = rois[0]

        roi_id = roi['id']
        cell_type = 0

        roi_marks = []
        cell_marks = []

        from cyborg.modules.ai.libs.algorithms.FISH_deployment import cal_fish_tissue
        nucleus_coords, red_signal_coords, green_signal_coords = cal_fish_tissue.run_fish(slide=task.slide_path)
        count_summary_dict = {
            'nuclues_num': len(nucleus_coords),
            'red_signal_num': len(red_signal_coords),
            'green_signal_num': len(green_signal_coords)
        }

        for nucleus_coord in nucleus_coords:
            cell_type = 0
            group_id = nucleus_group_id
            if len(nucleus_coord[:, 0]) > 3:
                this_mark = Mark(
                    position={'x': nucleus_coord[:, 0].tolist(), 'y': nucleus_coord[:, 1].tolist()},
                    method='freepen',
                    stroke_color=type_color_dict[cell_type],
                    mark_type=2,
                    diagnosis={'type': cell_type},
                    radius=1 / mpp,
                    area_id=roi_id,
                    group_id=group_id)
                cell_marks.append(this_mark)

        for red_signal_coord in red_signal_coords:
            cell_type = 1
            group_id = red_group_id
            this_mark = Mark(
                position={'x': [float(red_signal_coord[0])], 'y': [float(red_signal_coord[1])]},
                method='spot',
                stroke_color=type_color_dict[cell_type],
                mark_type=2,
                diagnosis={'type': cell_type},
                radius=4 * 1 / mpp,
                area_id=roi_id,
                group_id=group_id
            )
            cell_marks.append(this_mark)

        for green_signal_coord in green_signal_coords:
            cell_type = 2
            group_id = green_group_id
            this_mark = Mark(
                position={'x': [float(green_signal_coord[0])], 'y': [float(green_signal_coord[1])]},
                method='spot',
                stroke_color=type_color_dict[cell_type],
                mark_type=2,
                diagnosis={'type': cell_type},
                radius=4 * 1 / mpp,
                area_id=roi_id,
                group_id=group_id
            )
            cell_marks.append(this_mark)

            count_summary_dict['whole_slide'] = whole_slide

        roi_mark = Mark(
            id=roi_id,
            position={'x': [], 'y': []},
            mark_type=3,
            ai_result=count_summary_dict,
            diagnosis={'type': cell_type},
            radius=1 / mpp,
            stroke_color='grey',
            is_export=1,
            editable=1,
            group_id=group_name_to_id.get('ROI')
        )
        roi_marks.append(roi_mark)

        ai_suggest = '细胞核：{} 绿色信号点：{} 红色信号点：{}'.format(
            count_summary_dict.get('nuclues_num'),
            count_summary_dict.get('green_signal_num'),
            count_summary_dict.get('red_signal_num')
        )
        return ALGResult(
            ai_suggest=ai_suggest,
            roi_marks=roi_marks,
            cell_marks=cell_marks
        )

    def run_bm(self, task: AITaskEntity, group_name_to_id: dict):
        cells = []

        slide = open_slide(task.slide_path)
        mpp = slide.mpp if slide.mpp is not None else 0.242042

        from cyborg.modules.ai.libs.algorithms.BM.bm_alg import BM1115
        bboxes, scores, labels = BM1115().cal_bm(slide)

        roi = task.rois[0] if task.rois else task.new_default_roi()
        roi_id = roi['id']

        roi_marks = []
        cell_marks = []
        for idx in range(bboxes.shape[0]):
            box, score, label = map(int, list(bboxes[idx])), float(scores[idx]), int(labels[idx])
            xmin, ymin, xmax, ymax = box
            label_name = BMConsts.bm_class_list[label]
            BMConsts.label2_num_dict[label_name] += 1
            BMConsts.label1_num_dict[BMConsts.label_map_dict[label_name]] += 1
            if BMConsts.label2_num_dict[label_name] <= 100:
                BMConsts.label2_data_dict[label_name].append({
                    "id": idx,
                    "path": {"x": [xmin, xmax, xmax, xmin], "y": [ymin, ymin, ymax, ymax]},
                    "image": 0,
                    "editable": 0,
                    "dashed": 0,
                    "fillColor": "",
                    "mark_type": 2,
                    "area_id": roi_id,
                    "method": "rectangle",
                    "strokeColor": "red",
                    "radius": 0,
                    "cell_pos_prob": float(score)
                })

                cell_mark = Mark(
                    id=idx,
                    position={'x': [xmin, xmax, xmax, xmin], 'y': [ymin, ymin, ymax, ymax]},
                    stroke_color='red',
                    mark_type=2,
                    method='rectangle',
                    diagnosis={'type': label},
                    radius=1 / mpp,
                    area_id=roi_id,
                    editable=0,
                    group_id=group_name_to_id.get(BMConsts.bm_class_list[label]))
                cell_marks.append(cell_mark)

        for label1, num in sorted(BMConsts.label1_num_dict.items(), key=lambda k: k[1], reverse=True):
            label2_list = BMConsts.reversed_label_map_dict[label1]
            subset_label2_dict = {k: v for k, v in BMConsts.label2_num_dict.items() if k in label2_list}
            tier2_list = [{'label': k, 'num': v, 'data': BMConsts.label2_data_dict[k]} for (k, v) in
                          sorted(subset_label2_dict.items(), key=lambda k: k[1], reverse=True)]
            cells.append({'label': label1, 'num': num, 'data': tier2_list})

        ai_result = {
            'cell_num': bboxes.shape[0],
            'clarity': 1,
            'slide_quality': 1,
            'diagnosis': '',
            'cells': cells,
            'whole_slide': 1
        }

        roi_mark = Mark(
            id=roi_id,
            position={'x': [], 'y': []},
            mark_type=3,
            method='rectangle',
            is_export=1,
            ai_result=ai_result,
            stroke_color='grey',
            radius=1 / mpp,
            group_id=group_name_to_id.get('ROI')
        )
        roi_marks.append(roi_mark)

        return ALGResult(
            ai_suggest='',
            roi_marks=roi_marks,
            cell_marks=cell_marks
        )

    def get_ai_task_result(
            self, case_id: int, file_id: int, ai_type: AIType, task_id: Optional[int] = None
    ) -> Tuple[str, Optional[dict]]:
        if task_id is not None:
            task = self.repository.get_ai_task_by_id(task_id=task_id)
        else:
            task = self.repository.get_latest_ai_task(case_id=case_id, file_id=file_id, ai_type=ai_type)
        if not task:
            return '', {'done': True, 'rank': -2}

        if task.status in (AITaskStatus.success, AITaskStatus.failed):
            return '', {'done': True, 'rank': -1}
        else:
            if task.result_id:
                try:
                    result = AsyncResult(task.result_id, app=celery_app)
                    if result.ready() and result.get(timeout=0.001):
                        return '', {'done': True, 'rank': -1}
                except CeleryTimeoutError:
                    pass
                except Exception as e:
                    logger.exception(e)
                    self.update_ai_task(task, status=AITaskStatus.failed)
                    return 'AI处理发生异常', {'done': True, 'rank': -1}

            if task.status == AITaskStatus.analyzing:
                return '', {'done': False, 'rank': 0}
            else:
                start_id = cache.get(self.RANK0_TASK_ID_CACHE_KEY)
                ranking = self.repository.get_ai_task_ranking(task.id, start_id=start_id)
                if ranking is not None:
                    return '', {'done': False, 'rank': ranking + 1}

                return '', {'done': True, 'rank': -2}

    def get_analyze_threshold(self, params: dict, slices: List[dict]) -> dict:
        tbs_dict = TCTConsts.tct_multi_wsi_cls_dict.copy()

        result_ratio = {'NILM': 0, 'ASC-US': 0, 'LSIL': 0, 'ASC-H': 0, 'HSIL': 0, 'AGC': 0}
        result_count = {'NILM': 0, 'ASC-US': 0, 'LSIL': 0, 'ASC-H': 0, 'HSIL': 0, 'AGC': 0, 'sum': 0}
        result_fn = {'NILM': 0, 'ASC-US': 0, 'LSIL': 0, 'ASC-H': 0, 'HSIL': 0, 'AGC': 0, 'sum': 0}
        result_idx_dict = {'NILM': [], 'ASC-US': [], 'LSIL': [], 'ASC-H': [], 'HSIL': [], 'AGC': []}
        prob_list = []
        num_pos_cell_list = []
        num_manual = 0

        tct_probs = self.repository.get_tct_probs_by_slices(slices)

        for idx, tct_prob in enumerate(tct_probs):
            check_result = tct_prob.check_result
            prob_list.append(tct_prob.to_list())
            num_pos_cell_list.append((tct_prob.num_pos_cell or 0))
            check_result = '' if check_result is None else check_result
            check_result = check_result.split(' ')
            if len(check_result) >= 1:
                manual_diagnosis, manual_tbs = check_result[0], ''
                for tbs in sorted(tbs_dict.items(), key=lambda kv: kv[1])[::-1]:
                    # in case of check_result has more than one tbs label, use the higher class as the tbs label
                    if tbs[0] in check_result:
                        manual_tbs = tbs[0]
                        break
            else:
                continue

            if manual_diagnosis == '阳性':
                if manual_tbs.strip() in result_idx_dict:
                    result_idx_dict[manual_tbs].append(idx)
                else:
                    result_idx_dict['ASC-US'].append(idx)
            elif manual_diagnosis == '阴性':
                result_idx_dict['NILM'].append(idx)
            elif manual_diagnosis == '不确定':
                pass
            else:
                continue
            num_manual += 1

        num_slide = len(prob_list)
        if num_slide == 0:
            return {
                'ratio': result_ratio, 'count': result_count, 'fn': result_fn,
                'num_slide': num_slide, 'num_manual': num_manual, 'sensitivity': 0.0,
                'sensitivity_plus': 0.0, 'specificity': 0.0
            }

        prob_np = np.array(prob_list)
        num_pos_cell_np = np.array(num_pos_cell_list)
        pred_tbs = np.argmax(prob_np, axis=1)
        if params.get('threshold_range') == 1:
            pos_idx = np.where(np.logical_and(1 - prob_np[:, 0] > params.get('threshold_value'),
                                              num_pos_cell_np > params.get('min_pos_cell')))[0]
        else:
            # remain lsil, hsil, agc prediction
            pos_idx = np.where(np.any(np.vstack([
                1 - prob_np[:, 0] > params.get('threshold_value'),
                pred_tbs == tbs_dict['LSIL'],
                pred_tbs == tbs_dict['HSIL'],
                pred_tbs == tbs_dict['AGC']]), axis=0))[0]

        num_pos, num_pos_fn = 0, 0
        num_neg, num_neg_fn = 0, 0
        for k, v in tbs_dict.items():
            if k == 'NILM':
                this_label_num = int(num_slide - pos_idx.size)
                fn_num = int(np.intersect1d(pos_idx, np.array(result_idx_dict[k])).size)  # type: ignore
                num_neg, num_neg_fn = len(result_idx_dict[k]), fn_num
            elif k == 'ASC-US':
                this_label_num = int((pred_tbs[pos_idx] <= tbs_dict[k]).sum())  # type: ignore
                fn_num = int(np.setdiff1d(np.array(result_idx_dict[k]), pos_idx).size)
                num_pos += len(result_idx_dict[k])
                num_pos_fn += fn_num
            else:
                this_label_num = int((pred_tbs[pos_idx] == tbs_dict[k]).sum())  # type: ignore
                fn_num = int(np.setdiff1d(np.array(result_idx_dict[k]), pos_idx).size)
                num_pos += len(result_idx_dict[k])
                num_pos_fn += fn_num

            result_ratio[k] = round(this_label_num / num_slide, 4)
            result_count[k] = this_label_num
            result_fn[k] = fn_num

        sensitivity = round(1 - num_pos_fn / max(num_pos, 1), 4)  # 1- fn/P
        sensitivity_plus = round(
            1 - (num_pos_fn - result_fn['ASC-US']) / max(num_pos - len(result_idx_dict['ASC-US']), 1),
            4)
        specificity = round(1 - num_neg_fn / max(num_neg, 1), 4)  # 1 - fp/N

        result_count['sum'] = num_slide
        result_fn['sum'] = num_pos_fn

        return {
            'ratio': result_ratio, 'count': result_count, 'fn': result_fn,
            'num_slide': num_slide, 'num_manual': num_manual, 'sensitivity': sensitivity,
            'sensitivity_plus': sensitivity_plus, 'specificity': specificity
        }

    def hack_slide_quality(self, slice_info: dict) -> Tuple[Optional[str], str, str]:
        ai_suggest = slice_info['ai_suggest']
        ai_suggest_dict = ALGResult.parse_ai_suggest(ai_suggest)
        if ai_suggest_dict['flag'] == '0':
            return '无效的AI建议', '', ''

        diagnosis = ai_suggest_dict['diagnosis']
        dna_diagnosis = ai_suggest_dict["dna_diagnosis"]
        microbe = ai_suggest_dict["microbe"]

        if "-样本不满意" in ai_suggest:
            new_slide_quality = '1'
            diagnosis[1] = diagnosis[1].replace("-样本不满意", "")
        else:
            new_slide_quality = '0'
            diagnosis[1] = diagnosis[1] + "-样本不满意"

        new_ai_suggest = ' '.join(diagnosis) + ' ' + ','.join(microbe)
        if ';' in ai_suggest:
            new_ai_suggest = new_ai_suggest + ';' + dna_diagnosis

        return None, new_ai_suggest, new_slide_quality

    @transaction
    def hack_ai_suggest(
            self, diagnosis: str, microbe_list: List[int], slice_info: dict) -> Optional[str]:

        slice_id = slice_info['id']
        ai_suggest = slice_info['ai_suggest']

        ai_suggest_dict = ALGResult.parse_ai_suggest(ai_suggest)
        diagnosis_str = ' '.join(ai_suggest_dict['diagnosis'])
        microbe_str = ','.join(ai_suggest_dict['microbe'])
        dna_diagnosis = ai_suggest_dict['dna_diagnosis']

        diagnosis_type = TCTDiagnosisType.get_by_value(int(diagnosis)) if diagnosis else None
        microbe_types = list(filter(None, [MicrobeType.get_by_value(microbe) for microbe in microbe_list]))

        if diagnosis_type:
            prob_nilm, prob_hsil, prob_asch, prob_lsil, prob_ascus, prob_agc = {
                TCTDiagnosisType.HSIL: (0, 1, 0, 0, 0, 0),
                TCTDiagnosisType.ASC_H: (0, 0, 1, 0, 0, 0),
                TCTDiagnosisType.LSIL: (0, 0, 0, 1, 0, 0),
                TCTDiagnosisType.ASC_US: (0, 0, 0, 0, 1, 0),
                TCTDiagnosisType.AGC: (0, 0, 0, 0, 0, 1),
                TCTDiagnosisType.negative: (2, 0, 0, 0, 0, 0)
            }[diagnosis_type]
            tct_prob = self.repository.get_tct_prob(slice_id=slice_id)
            if tct_prob:
                tct_prob.update_data(
                    prob_nilm=prob_nilm,
                    prob_hsil=prob_hsil,
                    prob_asch=prob_asch,
                    prob_lsil=prob_lsil,
                    prob_ascus=prob_ascus,
                    prob_agc=prob_agc
                )
                self.repository.save_tct_prob(tct_prob)

            diagnosis_str = diagnosis_type.desc
            if '-样本不满意' in ai_suggest:
                diagnosis_str = f'{diagnosis_str}-样本不满意'

        if microbe_types:
            microbe_str = ','.join([microbe_type.desc for microbe_type in microbe_types])

        new_ai_suggest = f'{diagnosis_str} {microbe_str}'
        if dna_diagnosis:
            new_ai_suggest += ';' + dna_diagnosis

        ai_type = AIType.get_by_value(slice_info['alg'])
        current_date = slice_info['update_time'].split("T")[0]
        logger.info(f"{ai_type.value}-{slice_info['company']}-{current_date}")
        stats_list = self.repository.get_ai_stats(ai_type=ai_type, company=slice_info['company'], date=current_date)
        stats = stats_list[0] if stats_list else None
        if stats:
            if '阴性' in ai_suggest and '阳性' in new_ai_suggest:  # 阴性改成阳性
                stats.update_data(
                    negative_count_dr=stats.negative_count_dr - 1,
                    positive_count_dr=stats.positive_count_dr + 1
                )
            elif '阳性' in ai_suggest and '阴性' in new_ai_suggest:  # 阳性改成阴性
                stats.update_data(
                    negative_count_dr=stats.negative_count_dr + 1,
                    positive_count_dr=stats.positive_count_dr - 1
                )

            self.repository.save_ai_stats(stats=stats)

        logger.info('>>>>>>>>>>')
        logger.info(new_ai_suggest)

        return new_ai_suggest
