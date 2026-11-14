"""Lossless spelling normalization; never guess geometry or confidence."""
import copy


def normalize_thermal_output(output):
    normalized=copy.deepcopy(output);changes=[]
    if not isinstance(normalized,dict) or not isinstance(normalized.get('detections'),list):
        return normalized,changes
    aliases={'xmin':'x_min','ymin':'y_min','xmax':'x_max','ymax':'y_max'}
    for index,item in enumerate(normalized['detections']):
        if not isinstance(item,dict) or not isinstance(item.get('bbox'),dict):continue
        box=item['bbox']
        for alias,canonical in aliases.items():
            if alias not in box:continue
            if canonical in box and box[canonical]!=box[alias]:
                raise ValueError('Conflicting model bounding-box coordinate spellings')
            value=box.pop(alias);box[canonical]=value
            changes.append({'detection_index':index,'from':alias,'to':canonical,'value':value})
    return normalized,changes
