import json
import os


def save_prob_to_file(slide_path: str, result: dict, alg_name: str):
    if 'slide_pos_prob' in result:
        slide_pos_prob = result['slide_pos_prob'][0]
        slide_diagnosis = result['diagnosis']
        tbs_label = result['tbs_label']

        save_dict = {
            'slide_path': slide_path,
            'filename': os.path.basename(slide_path),
            'NILM': round(float(slide_pos_prob[0]), 5),
            'ASC-US': round(float(slide_pos_prob[1]), 5),
            'LSIL': round(float(slide_pos_prob[2]), 5),
            'ASC-H': round(float(slide_pos_prob[3]), 5),
            'HSIL': round(float(slide_pos_prob[4]), 5),
            'AGC': round(float(slide_pos_prob[5]), 5),
            'diagnosis': slide_diagnosis,
            'tbs_label': tbs_label
        }

        res_path = os.path.join(os.path.split(slide_path)[0])
        os.makedirs(res_path, exist_ok=True)

        with open(os.path.join(res_path, 'prob_{}.json'.format(alg_name)), 'w', encoding='utf-8') as f:
            json.dump(save_dict, f)

        return_dict = {k: save_dict[k] for k in ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']}
        return return_dict

    return None
