import os


def save_prob_to_file(slide_path: str, result: dict):
    if 'slide_pos_prob' in result:
        if result['slide_pos_prob'].shape == (6,):
            slide_pos_prob = result['slide_pos_prob']
        else:
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

        return_dict = {k: save_dict[k] for k in ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']}
        return return_dict

    return None
