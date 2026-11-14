"""Versioned thermal-label instructions; old paid outputs retain their identity."""
import hashlib

PROMPT_V1 = '''Inspect this ORIGINAL thermal frame for visible provisional mine-like ground signatures. Heat or circularity alone does not confirm a mine. Do not interpret people, animals, vegetation, broad warm ground, rocks or artifacts as mines. Return JSON {"detections":[{"label":"provisional mine-like thermal candidate","confidence":0.0,"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000},"evidence":"brief visible evidence","alternative":"other plausible explanation"}],"other_objects":[],"explanation":"brief visible evidence and limitations"}. Include ALL proposed mine-like candidates even if confidence is low. An empty detections array means no candidate is visible at ANY confidence. Named normalized boxes use 0..1000. Never invent geographic coordinates or confirm physical mines. No private chain-of-thought.'''

PROMPT_V2 = '''Inspect this ORIGINAL thermal drone frame for possible mine-like objects and localized ground signatures. Prioritize finding plausible candidates, including small, faint, low-contrast, partly obscured, blurred, or partially buried signatures, rather than only very obvious objects. Look for both hotter and colder anomalies, compact shapes, edges, rings, texture changes, and contrast with nearby ground; a candidate need not be circular or bright. Include a plausible subtle candidate with an appropriately LOWER confidence score. Calibrate confidence to the visible evidence: weak or ambiguous evidence should receive a low score, not an inflated score. Do not raise confidence to meet an acceptance threshold. For each candidate give a tight bounding box, brief visible evidence, and a plausible alternative explanation. Distinguish scene objects from sensor/reflection artifacts and out-of-focus camera equipment; list other visible objects separately without automatically dismissing nearby ground candidates. Return JSON {"detections":[{"label":"provisional mine-like thermal candidate","confidence":0.0,"bbox":{"x_min":0,"y_min":0,"x_max":1000,"y_max":1000},"evidence":"brief visible evidence","alternative":"other plausible explanation"}],"other_objects":[],"explanation":"brief visible evidence and limitations"}. Include ALL plausible proposed candidates even when confidence is low. An empty detections array means no plausible candidate was identified in this frame. Named normalized boxes use 0..1000. No private chain-of-thought.'''

PROMPT_VERSION = 'thermal-v2-recall-calibrated'
PROMPTS = {'thermal-v1-conservative': PROMPT_V1, PROMPT_VERSION: PROMPT_V2}


def prompt_metadata(text):
    if not isinstance(text, str):
        return {'label_prompt_version': 'unknown', 'label_prompt_sha256': None}
    version = next((name for name, value in PROMPTS.items() if value == text), 'unregistered')
    return {'label_prompt_version': version,
            'label_prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


def result_prompt_metadata(record, assumed_prompt=None):
    raw = record.get('raw_response') or {}
    text = raw.get('prompt') if isinstance(raw, dict) else None
    if isinstance(text, str):
        return prompt_metadata(text)
    version = record.get('label_prompt_version')
    if version in PROMPTS:
        expected = prompt_metadata(PROMPTS[version])
        if record.get('label_prompt_sha256') == expected['label_prompt_sha256']:
            return expected
    return prompt_metadata(assumed_prompt)


def registry():
    return {'active_version': PROMPT_VERSION,
            'versions': {name: {**prompt_metadata(text), 'prompt': text}
                         for name, text in PROMPTS.items()},
            'threshold': .6,
            'lower_confidence_policy': 'Retained as uncertain, never converted into confirmed negative backgrounds',
            'reuse_policy': 'Existing valid v1 labels remain v1; the prompt change alone does not trigger paid relabeling'}
